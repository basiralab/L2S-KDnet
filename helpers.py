def printTestResults(student_losses_G, teacher_losses_G, list_pagerankT, list_pagerankS, list_eigenvectorT,
                     list_eigenvectorS, list_student_loss_KL):
    print('=============================')
    print("End of the Program")
    print('=============================')
    print()

    print("Average Student Global Topology Evaluation Metric:")
    print(sum(student_losses_G) / 3)
    print(student_losses_G)
    print("Average Teacher Global Topology Evaluation Metric:")
    print(sum(teacher_losses_G) / 3)
    print(teacher_losses_G)

    print("Average Teacher PageRank Evaluation Metric:")
    print(sum(list_pagerankT) / 3)
    print(list_pagerankT)
    print("Average Student PageRank Evaluation Metric:")
    print(sum(list_pagerankS) / 3)
    print(list_pagerankS)

    print("Average Teacher EigenVector Evaluation Metric:")
    print(sum(list_eigenvectorT) / 3)
    print(list_eigenvectorT)
    print("Average Student EigenVector Evaluation Metric:")
    print(sum(list_eigenvectorS) / 3)
    print(list_eigenvectorS)

    print("Average Student KL Evaluation Metric:")
    print(sum(list_student_loss_KL) / 3)
    print(list_student_loss_KL)


def printFoldResults(fold, teacher_loss_G, student_loss_G, pagerankT, pagerankS, eigenvectorT, eigenvectorS,
                     student_loss_KL):
    print('=============================')
    print("End of the Fold" + str(fold))
    print('=============================')

    print("Fold" + str(fold) + " Teacher Global Topology Evaluation Metric: ", teacher_loss_G)

    print("Fold" + str(fold) + " Student Global Topology Evaluation Metric: ", student_loss_G)

    print("Fold" + str(fold) + " Teacher PageRank Evaluation Metric: ", pagerankT)

    print("Fold" + str(fold) + " Student PageRank Evaluation Metric: ", pagerankS)

    print("Fold" + str(fold) + " Teacher EigenVector Evaluation Metric: ", eigenvectorT)

    print("Fold" + str(fold) + " Student EigenVector Evaluation Metric: ", eigenvectorS)

    print("Fold" + str(fold) + " Student KL Evaluation Metric: ", student_loss_KL)
