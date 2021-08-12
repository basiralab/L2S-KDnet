import datetime
import itertools
import time

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error

from centrality import *
from data_loader import *
from model import *

matplotlib.use('Agg')


class L2S_KDnet_withoutLocalTopology(object):
    def __init__(self, real_morphological_loader, real_functional_loader, opts):

        self.real_morphological_loader = real_morphological_loader
        self.real_functional_loader = real_functional_loader
        self.criterionIdt = torch.nn.L1Loss()
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        """
        Build teachers and students and encoder and initialize optimizers.
        """
        ## Teacher model components
        self.encoder = GCNencoder(self.opts.in_feature_t0, self.opts.hidden1, self.opts.hidden2, self.opts.dropout).to(
            self.device)
        self.discriminator = Discriminator(self.opts.hidden2, 1, self.opts.dropout).to(self.device)
        self.generator = GCNdecoder(self.opts.hidden2, self.opts.hidden3, self.opts.SRout, self.opts.dropout).to(
            self.device)
        ## Student model
        self.Student = Student(self.opts.in_feature_t0, self.opts.hidden1, self.opts.hidden2, self.opts.hidden3,
                               self.opts.SRout,
                               self.opts.dropout).to(self.device)

        ## initializations of optimizers
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.opts.d_lr,
                                            [self.opts.beta1, self.opts.beta2])
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), self.opts.g_lr,
                                            [self.opts.beta1, self.opts.beta2])
        self.e_optimizer = torch.optim.Adam(self.encoder.parameters(), self.opts.g_lr,
                                            [self.opts.beta1, self.opts.beta2])
        param_list = [self.Student.parameters()]
        self.student_optimizer = torch.optim.Adam(itertools.chain(*param_list),
                                                  self.opts.s_lr, [self.opts.beta1, self.opts.beta2])

    def restore_model(self, resume_iters, model_name="teacher", fold=0):
        """
        Restore the trained students and encoder.
        """
        print('Loading the trained models from step {}...'.format(resume_iters))

        if model_name == "teacher":
            generator_path = os.path.join(self.opts.checkpoint_dir,
                                          'generator-{}-{}-{}-{}.ckpt'.format(fold,resume_iters, self.opts.local_topology_loss_weight,
                                                                                       self.opts.teacher_loss_weight))
            encoder_path = os.path.join(self.opts.checkpoint_dir,
                                        'encoder-{}-{}-{}-{}.ckpt'.format(fold, resume_iters,self.opts.local_topology_loss_weight, self.opts.teacher_loss_weight))

            self.generator.load_state_dict(torch.load(generator_path, map_location=lambda storage, loc: storage))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        else:
            Student_path = os.path.join(self.opts.checkpoint_dir,
                                        'student-{}-{}-{}-{}.ckpt'.format(fold, resume_iters,self.opts.local_topology_loss_weight, self.opts.teacher_loss_weight ))
            self.Student.load_state_dict(torch.load(Student_path, map_location=lambda storage, loc: storage))

    def reset_grad(self):
        """
        Reset the gradient buffers.
        """
        self.student_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        self.e_optimizer.zero_grad()

    def show_mtrx(self, m, name):
        """
        Visualize matrix and save.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.matshow(m, cmap=plt.cm.seismic)
        fig.colorbar(cax)
        plot_path = os.path.join(self.opts.plot_dir,
                                 "{}_{}_{}.png".format(name, self.opts.local_topology_loss_weight, self.opts.teacher_loss_weight))
        plt.savefig(plot_path)
        # plt.show()

    def any_loss(self, real, predicted, metric):
        """
        Compute KL, MAE losses.
        """
        self.MAE = torch.nn.L1Loss()
        self.KL = nn.KLDivLoss()
        if metric == 'KL':
            m = nn.LogSoftmax(dim=1)
            predicted = m(predicted)
            return self.KL(predicted, real)
        elif metric == 'MAE':
            return self.MAE(real, predicted)
        elif metric == 'global_topology':
            return self.MAE(real, predicted)
        elif metric == 'local_topology':
            if real.shape[1] == 595:  # feature vector -> 595 extracted from 35x35
                size = 35
            else:                     # feature vector -> 12720 extracted from 160x160
                size = 160

            # Local topology loss
            real_topology = topological_measures(real, size)
            fake_topology = topological_measures(predicted, size)

            # 0:Eigenvector    1:PageRank
            return torch.tensor(
                mean_absolute_error(fake_topology[0], real_topology[0]), requires_grad=True), torch.tensor(
                mean_absolute_error(fake_topology[1], real_topology[1]), requires_grad=True)
        elif metric == 'eigenvector':
            if real.shape[1] == 595:  # feature vector -> 595 extracted from 35x35
                size = 35
            else:  # feature vector -> 12720 extracted from 160x160
                size = 160

            # Local topology loss
            real_topology = topological_measures(real, size, True)
            fake_topology = topological_measures(predicted, size, True)

            # 0:Eigenvector
            return torch.tensor(mean_absolute_error(fake_topology[0], real_topology[0]), requires_grad=True)
    def train(self, fold):
        """
        Train TS
        """
        t0_iter_M = iter(self.real_morphological_loader)
        t0_iter_F = iter(self.real_functional_loader)

        morph_iters = []
        for loader in self.real_morphological_loader:
            morph_iters.append(iter(loader))

        func_iters = []
        for loader in self.real_functional_loader:
            func_iters.append(iter(loader))

        # Start training from scratch or resume training.
        start_iters = 0
        if self.opts.resume_iters:
            start_iters = self.opts.resume_iters
            self.restore_model(self.opts.resume_iters, fold)

        # Start training.
        print("********-------------*********")
        print('Super-resolution TS network...')
        print("********-------------*********")
        start_time = time.time()

        loss_students_plot = []
        loss_embedding_plot = []

        generator_loss = []
        discriminator_loss = []

        print("################################")
        print(" 1. Train the Teacher")
        print("################################")
        for i in range(start_iters, self.opts.num_iters):
            print("-------------iteration-{}-------------".format(i))
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            try:
                t0_morph_encoder_T = next(t0_iter_M)
            except:
                t0_iter_M = iter(self.real_morphological_loader)
                t0_morph_encoder_T = next(t0_iter_M)
            t0_M_encoder_T = t0_morph_encoder_T[0].to(self.device)

            try:
                t0_func_encoder_T = next(t0_iter_F)
            except:
                t0_iter_F = iter(self.real_functional_loader)
                t0_func_encoder_T = next(t0_iter_F)
            t0_F_encoder_T = t0_func_encoder_T[0].to(self.device)

            # =================================================================================== #
            #                           2. Train generators                                       #
            # =================================================================================== #
            print("-------------Train the generators-------------")

            adj = torch.eye(t0_M_encoder_T.shape[0]).to(self.device)
            embedding = self.encoder(t0_M_encoder_T, adj)

            adj = torch.eye(embedding.shape[0]).to(self.device)
            super_resolved_matrix = self.generator(embedding, adj)

            sigmoid, softmax = self.discriminator(embedding, adj)

            g_loss_adversarial = F.binary_cross_entropy_with_logits(sigmoid, torch.ones_like(sigmoid))

            teacher_loss = self.any_loss(t0_F_encoder_T, super_resolved_matrix, "global_topology")

            g_loss = g_loss_adversarial + (self.opts.teacher_loss_weight * teacher_loss)

            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            self.e_optimizer.step()

            print("Generator Loss: ", g_loss.item())
            generator_loss.append(g_loss.item())

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            (mu, sigma) = norm.fit(t0_F_encoder_T.detach())
            dist = np.random.normal(mu, sigma, (279, 50))
            real_dist = torch.Tensor(dist)

            adj = torch.eye(dist.shape[0]).to(self.device)
            sigmoid_real, softmax_real = self.discriminator(real_dist, adj)

            adj = torch.eye(embedding.shape[0]).to(self.device)
            sigmoid_fake, softmax_fake = self.discriminator(embedding, adj)

            print("Train the discriminator")

            real_loss = F.binary_cross_entropy_with_logits(sigmoid_real,
                                                           (torch.ones_like(sigmoid_real, requires_grad=False)))
            fake_loss = F.binary_cross_entropy_with_logits(sigmoid_fake.detach(),
                                                           torch.zeros_like(sigmoid_fake))

            d_loss = (real_loss + fake_loss) / 2

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            print("Discriminator Loss: ", d_loss.item())
            discriminator_loss.append(d_loss.item())

            # =================================================================================== #
            #                                 3. Miscellaneous                                    #
            # =================================================================================== #
            # print out training information.
            if (i + 1) % self.opts.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.opts.num_iters)
                """for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                """
                print(log)

            # save model checkpoints.
            if (i + 1) % self.opts.model_save_step == 0:
                generator_path = os.path.join(self.opts.checkpoint_dir,
                                              'generator-{}-{}-{}-{}.ckpt'.format(fold,
                                                                                           i + 1, self.opts.local_topology_loss_weight,
                                                                                           self.opts.teacher_loss_weight))
                torch.save(self.generator.state_dict(), generator_path)

                encoder_path = os.path.join(self.opts.checkpoint_dir,
                                            'encoder-{}-{}-{}-{}.ckpt'.format(fold, i + 1,self.opts.local_topology_loss_weight,
                                                                                       self.opts.teacher_loss_weight))
                torch.save(self.encoder.state_dict(), encoder_path)

                print('Saved model checkpoints into {}...'.format(self.opts.checkpoint_dir))

                print('=============================')
                print("End of Training Teacher")
                print('=============================')

            if i == (self.opts.num_iters - 1):
                # =================================================================================== #
                #      Restore the trained teacher  from the last iteration to train the Student      #
                # =================================================================================== #
                self.restore_model(self.opts.test_iters, model_name="teacher", fold=fold)
                self.encoder.eval()
                self.generator.eval()

                with torch.no_grad():
                    adj = torch.eye(t0_M_encoder_T.shape[0]).to(self.device)
                    embedding = self.encoder(t0_M_encoder_T, adj)

                    adj = torch.eye(embedding.shape[0]).to(self.device)
                    predicted_F_teacher = self.generator(embedding, adj)

                print("################################")
                print(" 3. Train the Student ")
                print("################################")
                for j in range(start_iters, self.opts.num_iters):
                    print("-------------iteration{}-------------".format(j))
                    # =================================================================================== #
                    #                                 Train the Student                                   #
                    # =================================================================================== #
                    student_loss = 0
                    adj = torch.eye(t0_M_encoder_T.shape[0]).to(self.device)
                    embedding_student, predicted_F_student = self.Student(t0_M_encoder_T, adj)

                    adj = torch.eye(embedding_student.shape[0]).to(self.device)
                    predict_decoderof_t_usingembed_s = self.generator(embedding_student, adj)

                    student_loss_1 = self.any_loss(predicted_F_student, predicted_F_teacher, "global_topology")
                    student_loss_2 = self.any_loss(embedding_student, embedding, "global_topology")
                    student_loss_3 = self.any_loss(predict_decoderof_t_usingembed_s, predicted_F_teacher,
                                                   "global_topology")
                    student_loss = (student_loss_1 + student_loss_2 + student_loss_3) / 3

                    print("Student Loss: ", student_loss.detach().item())

                    self.reset_grad()
                    student_loss.backward()
                    self.student_optimizer.step()

                    # Logging.
                    loss = {'Student/loss': student_loss.item()}
                    loss_students_plot.append(student_loss)
                    loss_embedding_plot.append(student_loss_2)

                    # save model checkpoints.
                    if (j + 1) % self.opts.model_save_step == 0:
                        Student_path = os.path.join(self.opts.checkpoint_dir,
                                                    'student-{}-{}-{}-{}.ckpt'.format(fold,i + 1, self.opts.local_topology_loss_weight,
                                                                                               self.opts.teacher_loss_weight))
                        torch.save(self.Student.state_dict(), Student_path)
                        print('Saved model checkpoints into {}...'.format(self.opts.checkpoint_dir))

                    if j == (self.opts.num_iters - 1):
                        plt.figure(1)
                        losses = []
                        epochs = []
                        counter = 1
                        for i in generator_loss:
                            epochs.append(counter)
                            losses.append(i)
                            counter += 1
                        fig = plt.figure()
                        plt.plot(epochs, losses, '-b', label="Generator")
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        plot_path = os.path.join(self.opts.plot_dir,
                                                 "generatorLoss_{}_{}_{}.png".format(
                                                    fold,self.opts.local_topology_loss_weight, self.opts.teacher_loss_weight))
                        plt.savefig(plot_path, bbox_inches='tight')

                        plt.figure(2)
                        fig = plt.figure()
                        losses = []
                        epochs = []
                        counter = 1
                        for i in discriminator_loss:
                            epochs.append(counter)
                            losses.append(i)
                            counter += 1
                        plt.plot(epochs, losses, '-r', label="Discriminator")
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        plot_path = os.path.join(self.opts.plot_dir,
                                                 "discriminatorLoss_{}_{}_{}.png".format(
                                                     fold,self.opts.local_topology_loss_weight, self.opts.teacher_loss_weight))
                        plt.savefig(plot_path, bbox_inches='tight')

                    if j == (self.opts.num_iters - 1):
                        plt.figure(3)
                        losses = []
                        epochs = []
                        counter = 1
                        for i in loss_students_plot:
                            epochs.append(counter)
                            losses.append(i.detach().item())
                            counter += 1
                        fig = plt.figure()
                        plt.plot(epochs, losses, '-g', label="Student")
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        plot_path = os.path.join(self.opts.plot_dir,
                                                 "studentLoss_{}_{}_{}.png".format(
                                                     fold,self.opts.local_topology_loss_weight, self.opts.teacher_loss_weight))
                        plt.savefig(plot_path, bbox_inches='tight')

                        print('============================================')
                        print("End of Training the Student")
                        print('============================================')

    def test(self, real_morphological_loader, real_functional_loader, i):
        """
        Test the trained Students on another dataset.
        """
        print("################################################################################")
        print(" 2. Restore the trained student from the last iteration to test the Student ")
        print("################################################################################")
        self.restore_model(self.opts.test_iters, model_name="teacher", fold=i)
        self.encoder.eval()
        self.generator.eval()

        self.restore_model(self.opts.test_iters, model_name="student", fold=i)
        self.Student.eval()

        t0_iter_M = iter(real_morphological_loader)
        t0_iter_F = iter(real_functional_loader)

        try:
            t0_morph_encoder_T = next(t0_iter_M)
        except:
            t0_iter_M = iter(real_morphological_loader)
            t0_morph_encoder_T = next(t0_iter_M)
        t0_M_encoder_T = t0_morph_encoder_T[0].to(self.device)

        try:
            t0_func_encoder_T = next(t0_iter_F)
        except:
            t0_iter_F = iter(real_functional_loader)
            t0_func_encoder_T = next(t0_iter_F)

        t0_F_encoder_T = t0_func_encoder_T[0].to(self.device)

        with torch.no_grad():
            adj = torch.eye(t0_M_encoder_T.shape[0]).to(self.device)
            embedding_teacher = self.encoder(t0_M_encoder_T, adj)

            adj = torch.eye(embedding_teacher.shape[0]).to(self.device)
            predicted_F_teacher = self.generator(embedding_teacher, adj)

            embedding_student, predicted_F_student = self.Student(t0_M_encoder_T, adj)

        student_loss_KL = self.any_loss(t0_F_encoder_T, predicted_F_student, "KL")
        print("Student Evaluation KL: ", student_loss_KL.item())

        eigenvectorS, pagerankS = self.any_loss(t0_F_encoder_T, predicted_F_student, "local_topology")

        eigenvectorT, pagerankT = self.any_loss(t0_F_encoder_T, predicted_F_teacher, "local_topology")

        student_loss_G = self.any_loss(t0_F_encoder_T, predicted_F_student, "global_topology")
        print("Student Evaluation Global: ", student_loss_G.item())

        teacher_loss_G = self.any_loss(t0_F_encoder_T, predicted_F_teacher, "global_topology")
        print("Teacher Evaluation Global: ", teacher_loss_G.item())

        print('=============================')
        print("End of Testing both networks")
        print('=============================')

        # =================================================================================== #
        #                          Save results of both networks                              #
        # =================================================================================== #
        print("saving TS prediction into csv file...")
        f = predicted_F_teacher.detach().cpu().numpy()
        dataframe = pd.DataFrame(data=f.astype(float))
        dataframe.to_csv(
            r'%s/results/teacher_predicted_functional_graphs_%d_%f_%f.csv' % (
                self.opts.result_root, i, self.opts.local_topology_loss_weight, self.opts.teacher_loss_weight), sep=' ', header=True,
            float_format='%.6f', index=False)

        print("saving TS prediction into csv file...")
        f = predicted_F_student.detach().cpu().numpy()
        dataframe = pd.DataFrame(data=f.astype(float))
        dataframe.to_csv(
            r'%s/results/student_predicted_functional_graphs_%d_%f_%f.csv' % (self.opts.result_root,
                                                                                       i, self.opts.local_topology_loss_weight,
                                                                                       self.opts.teacher_loss_weight), sep=' ',
            header=True,
            float_format='%.6f', index=False)

        return eigenvectorS.item(), pagerankS.item(), \
               eigenvectorT.item(), pagerankT.item(), \
               student_loss_G.item(), teacher_loss_G.item(), \
               student_loss_KL.item()
