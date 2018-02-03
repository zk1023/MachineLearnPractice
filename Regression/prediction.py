from bs4 import BeautifulSoup
import numpy as np
import Regression.towardRegre as toward
import Regression.redigReg as redig
import Regression.regressions as regression
import random
from sklearn import linear_model

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    #根据html页面结构进行解析
    currentRow = soup.find_all('table', r = "%d" % i)
    while len(currentRow) != 0:
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        if(lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        #查找是否已经标志出售，直收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品#%d 没有出售" % i)
        else:
            #解析页面，获取价格
            soldPrice  = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            if sellingPrice > origPrc *0.5:
                print("%d\t%d\t%d\t%f\t%f\t" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r = "%d" % i)
def setDataCollect(retX, retY):
    #2006年的乐高8288 部件数目800， 原价49.99
    scrapePage(retX, retY, 'lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'lego10196.html', 2009, 3263, 249.99)

#使用线性回归
def useStandRegres():
    lgX = []; lgY = []
    setDataCollect(lgX, lgY)
    data_num, features_num = np.shape(lgX)
    print(data_num)
    print(features_num)
    lgX1 = np.mat(np.ones((data_num, features_num + 1)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = regression.standRegres(lgX1, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (ws[0], ws[1],ws[2], ws[3], ws[4]))

#使用岭回归
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = redig.ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX  = (matTestX - meanTrain) / varTrain
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)
            errorMat[i, k] = toward.rssError(yEst.T.A, np.array(testY))
    meanErrors = np.mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX, unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0, 3]))
    # print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (
    # (-1 * np.sum(np.multiply(meanX, unReg)) + np.mean(yMat)), unReg[0, 0], unReg[0, 1], unReg[0, 2], unReg[0, 3]))

def usesklearn():
    reg = linear_model.Ridge(alpha=.5)
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    reg.fit(lgX, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]))

if __name__ == '__main__':
    # useStandRegres()

    # lgX = []
    # lgY = []
    # setDataCollect(lgX, lgY)
    # crossValidation(lgX, lgY)

    # lgX = []
    # lgY = []
    # setDataCollect(lgX, lgY)
    # print(redig.ridgeTest(lgX, lgY))
    usesklearn()
