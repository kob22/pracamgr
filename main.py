from data import importdata
from runclassifiers import svm

rettreeall = []
retbaggingall = []

german = importdata.importfile("files/german.data-numeric",1000,20)
# rettree = tree.runtree(german.data,german.target)
#
# rettreeall.append(reduce(lambda x, y: x + y, rettree) / len(rettree))
#
# retbagging = bagging.runbaggin(german.data, german.target)
# retbaggingall.append(reduce(lambda x, y: x + y, retbagging) / len(retbagging))
#
# print(rettreeall)
# print(retbaggingall)

retsvclinearall = []
retsvcpolyall = []
retsvcrbfall = []
retsvcsigmoidall = []


for num in range(0, 2):
    #german.data, german.target = shuffle(german.data, german.target, random_state=13)
    german.shuffle2()
    #print(german)
    retsvclinear = svm.runsvc('linear', german.data, german.target, 10)
    print("SVML czas: %f, wynik: %s" % (retsvclinear[1], retsvclinear[0]))
    retsvclinearall.append(reduce(lambda x, y: x + y, retsvclinear[0]) / len(retsvclinear[0]))

    retsvcpoly = svm.runsvc('poly', german.data, german.target, 10)
    print("SVMP czas: %f, wynik: %s" % (retsvcpoly[1], retsvcpoly[0]))
    retsvcpolyall.append(reduce(lambda x, y: x + y, retsvcpoly[0]) / len(retsvcpoly[0]))

    retsvcrbf = svm.runsvc('rbf', german.data, german.target, 10)
    print("SVMS czas: %f, wynik: %s" % (retsvcrbf[1], retsvcrbf[0]))
    retsvcrbfall.append(reduce(lambda x, y: x + y, retsvcrbf[0]) / len(retsvcrbf[0]))


    retsvcsigmoid = svm.runsvc('sigmoid', german.data, german.target, 10)
    print("SVMS czas: %f, wynik: %s" % (retsvcsigmoid[1], retsvcsigmoid[0]))
    retsvcsigmoidall.append(reduce(lambda x, y: x + y, retsvcsigmoid[0]) / len(retsvcsigmoid[0]))







print(retsvclinearall)
print(retsvcpolyall)
print(retsvcrbf)
print(retsvcsigmoidall)

#tree.runtree(data,target)
#adaboost.runada(data,target)
#srandomforest.runforest(data,target)
