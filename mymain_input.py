'''
문항난이도/곤란도(item difficulty) - 1모수 모형

문항이 어느 능력 수준에서 기능하는가를 나타내는 지수
문항의 답을 맞힐 확률이 .5에 해당되는 능력 수준의 점을 의미한다.
category가 여러개인 경우 답을 맞힐 확률이 아닌 해당 category를 선택할 곤란도를 나타낸다.

문항변별도(item discrimination) - 2모수 모형

문항이 피험자를 능력에 따라 변별하는 정도를 나타내는 지수
문항반응이론에서 문항변별도란 문항특성곡선상의 문항의 답을 맞힐 확률이 .5에 해당하는 점에서 문항특성곡선의 기울기를 말함

문항추측도(item guessing) - 3모수 모형

능력이 전혀 없음에도 불구하고 문항의 답을 맞힐 확률
즉, -∞의 능력 수준을 가지고 있는 피험자가 문항의 답을 맞힐 확률을 문항추측도라 하며, 이를 c로 표기한다.
5지선다면 찍을 확률 1/5, c =0.2

정보함수 (information)
정보함수는 검사가 피험자들의 능력을 얼마나 정확하게 추정하였느냐하는 정보를 제공하는 것을 말합니다.
1) 문항정보함수는 측정오차와 관계있는 분산값, 즉 능력추정의 표준오차의 제곱에 반비례한다. 따라서 능력추정의 표준오차가 작을 수록 문항정보함수 값은 커집니다.
2) 문항정보함수는 문항변별도가 높을수록, 그리고 문항난이도와 능력수준이 일치할수록 문항정보함수가 커진다.
3) 문항반응이론에서는 피험자의 능력추정오차는 정보함수에 의하여 나타나므로 피험자 마다 다른 능력측정오차를 갖는다.
4) 고전검사이론의 신뢰도에 해당한다.


GRM(graded response model) 등급반응모형
for 리커트 척도


GPCM(generalized partial credit model) 일반화부분점수모형
for 부분 점수


위치모수(location parameter)문항난이도, β, b
문항이 능력 수준의 어느 지점에서 기능하는가
즉, 문항특성곡선 상에서 문항의 답을 맞힐 확률이 .5가 되는 능력 수준
위치모수가 오른쪽에 위치할수록 문항난이도가 높음
그 문제를 맞힐 확률이 .5가 되기 위해서 더 높은 능력이 필요하다는 이야기니까


​척도모수(scale parameter)문항변별도, α, a
문항특성곡선이 위치모수  위에 있는 수검자와 그 아래에 있는 수검자를 변별하는 정도
즉, 문항의 답을 맞힐 확률이 .5에 해당되는 능력 수준에서의 문항특성곡선의 접선의 기울기
다른말로 ICC가 더 완만할수록 문항변별도가 낮음

'''

import csv #input 문 이용하기 위함
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename_par = "./item.csv"
# filename_resp = "./response.csv"


item_par = pd.read_csv(filename_par, sep=",", header=None)
item_par = np.array(item_par)


# resp_data = pd.read_csv(filename_resp, sep=",", header=None)
# resp_data = np.array(resp_data)


# nExaminees = resp_data.shape[0] # 피험자 20명
nExaminees = 3 #임시
ni = item_par.shape[0] ############################################################## 문항수 10개


model = 1   # (문항반응이론)IRT모델 종류 1: GRM, 2:GPCM 
selection_method = 1 # 문제 제시 방법
interim_Theta = 1 # 1: EAP(베이지안 기대 사후 확률), 2: MLE(최대 우도법)
Show_Theta_Audit_Trail = 1 # 수험자 별 test - theta 그래프 보여주기

first_item_selection = 1 # 첫번째 문제 제시방법 (1일 때, start_theta = prior_mean)
first_at_theta = 0 # 기본 설정 theta
first_item = 1 # 기본 설정 문항


minTheta = -4 # 능력치 최소 -4
maxTheta = 4 # 능력치 최대 4
inc = 0.10 # 능력치 눈금
theta = np.arange(minTheta, maxTheta+inc, inc) #-4에서 4.1까지 0.1step을 가진 배열
ntheta = len(theta) # 81


# Prior Distribution(사전 분포)
prior_dist = 1 # 1 : Normal, 2: Logistic
prior_mean = 0 # 평균
prior_sd = 1 # 표준편차


# 정규분포 공식
normal_distribution = np.exp(-(theta.astype(float) / prior_sd) ** 2 / 2) / (2 * np.pi * prior_sd) ** .5


# 사전분포 = 정규분포
prior = normal_distribution


## Stopping Criteria
maxNI = 10
minNI = 5
maxSE = 0.3


items_used = np.full((nExaminees,maxNI), np.nan)
selected_item_resp = np.full((nExaminees, maxNI), np.nan)


ni_administered = np.zeros(nExaminees)


theta_CAT = np.full(nExaminees, np.nan) # nExaminees 배열을 NA로 채움
sem_CAT = np.full(nExaminees, np.nan) # standard error examinees
theta_history = np.full((nExaminees,maxNI), np.nan)
se_history = np.full((nExaminees,maxNI), np.nan)
posterior_matrix = np.full((nExaminees, ntheta), np.nan)
LH_matrix = np.full((nExaminees, ntheta), np.nan) #likelihood


#######################################################################


SE_method = 1 # 표준오차 계산방법 1: posterior, 2 : information
nitem = item_par.shape[0]  # 10
ncat = item_par[:, -1].astype('uint8')  # [5,5,5,5,5,5,5,5,5,5]
disc = item_par[:, 0]  # shape = (10)
beta = item_par[:, 1:-1]  # shape = (10,4)
ntheta = len(theta)
maxCat = ncat[0]
a = np.zeros(nitem)


################## previsional probability information #########################

def prep_prob_info():
    pp = np.zeros((maxCat, ntheta, nitem))
    matrix_info = np.zeros((ntheta, nitem))
    if model == 1: # GRM
        for i in range(nitem):
            ps = np.zeros((ntheta, maxCat + 1))
            ps[:, 0] = 1 
# ps = [1, b1, b2, b3, b4, 0] * 81
# b = beta값, 역치값이므로 이후 별도의 계산 필요

            for k in range(maxCat - 1):
                ps[:, k + 1] = 1 / (1 + np.exp(-1.7 * disc[i] * (theta - beta[i, k])))
# ps = [1, p(theta-b1), p(theta-b2), p(theta-b3), p(theta-b4)] * 81
# 정규분포를 적분한 2모수 정규오자이브모형은 계산이 어려우므로
# 대신 2모수 로지스틱모형을 사용(적분 역할)
# 두 모형의 차이를 문항변별도에 1.7을 곱해줌으로 없애준다
# 해석: 해당 theta에서 해당 답을 체크할 확률, p(theta)

            for k in range(maxCat):
                pp[k, :, i] = ps[:, k] - ps[:, k + 1]
                matrix_info[:, i] += (1.7 * disc[i] * (
                        ps[:, k] * (1 - ps[:, k]) - ps[:, k + 1] * (1 - ps[:, k + 1]))) ** 2 / pp[k, :, i]
# matrix_info = theta(1-p(theta-b1)), theta(p(theta-b1)-p(theta-b2)), theta(p(theta-b2)-p(theta-b3)), theta(p(theta-b3)-p(theta-b4)) * 81
# 해당 문제에서 체크되었을 때 가장 기울기(미분)가 큰 값 = p(theta)가 0.5인 theta값

    elif model == 2: # GPCM
        for i in range(nitem):
            cb = np.zeros(maxCat)
            cb[1:] = beta[i] # [0, b1, b2, b3, b4]
            zz = np.zeros((ntheta, maxCat)) # (81, 5)
            sdsum = 0
            den = np.zeros(ntheta) # 81

            for k in range(maxCat): # 5
                sdsum += cb[k] # beta값을 추가
                zz[:, k] = np.exp(1.7 * disc[i] * (k * theta - sdsum))
                # (5*theta) - (5category의 beta)합, 전체 category에서 
                den += zz[:, k]

            AX = np.zeros(ntheta)
            BX = np.zeros(ntheta)

            for k in range(maxCat):
                pp[k, :, i] = zz[:, k] / den
                AX += k ** 2 * pp[k, :, i]
                BX += BX + k * pp[k, :, i]

                matrix_info[:, i] = 1.7 ** 2 * disc[i] ** 2 * (AX - BX ** 2)

    return pp, matrix_info

#########################################################Generating Response ###

def genResp(nSimulee):
    popMean = 0
    popSD = 1
    dist = np.random.randn(nSimulee) * popSD + popMean  # popmean, popSD인 정규분포 생성
    resp = np.ones((nSimulee, nitem))  # origin : resp = np.ones((ntheta,ni))
    pp = np.zeros((maxCat, nSimulee, nitem))  # origin: pp = np.zeros((maxCat, ntheta, ni))

    if model == 1:  # GRM
        for i in range(nitem):
            # ncat = int(item_par[i, -1])
            ncat = int(maxCat)

            # a = item_par[i, 0]
            # cb = item_par[i, 1:-1]
            a = disc[i]
            cb = beta[i]

            ps = np.zeros((nSimulee, ncat + 1))  # origin : ps = np.zeros((ntheta,ncat+1))
            ps[:, 0] = 1
            
            for k in range(ncat - 1):
                ps[:, k + 1] = 1 / (1 + np.exp(-1.7 * a * (dist - cb[k])))

            # pp[0,:,i] = 1-ps[:,0] #이해가 안가는 코드. 그냥 0을 대입하는 것이랑 무슨차이지?
            pp[ncat - 1, :, i] = ps[:, ncat - 1]
            for k in range(ncat):
                # pp[,i,k]=ps[,k]-ps[,k+1]
                pp[k, :, i] = ps[:, k] - ps[:, k + 1]

    elif model == 2:  # GPCM
        for i in range(nitem):
            # ncat = int(item_par[i, -1])  # origin : ncat = Par.iloc[i, -1]
            ncat = int(maxCat)

            # a = item_par[i, 0]  # origin : a = Par.iloc[i,0]
            a = disc[i]

            cb = np.zeros(ncat)
            # cb[1:] = item_par[i, 1:-1]  # origin : cb = Par.iloc[i, 1:-1]
            cb[1:] = beta[i]

            zz = np.zeros((nSimulee, ncat))
            sdsum = 0
            den = np.zeros(nSimulee)

            for k in range(ncat):
                sdsum += cb[k]
                zz[:, k] = np.exp(1.7 * a * (k * dist - sdsum))
                den += zz[:, k]

            for k in range(ncat):
                pp[k, :, i] = zz[:, k] / den

    random = np.random.rand(nSimulee, nitem)
    for i in range(nitem):
        # ncat = item_par[i, -1].astype('int32')
        ncat = int(maxCat)

        sump = np.zeros(nSimulee)
        for k in range(ncat - 1):
            sump += pp[k, :, i]

            resp[:, i][random[:, i] > sump] += 1

    return resp, dist

#######################End Generating Response##################################

################# Calculation Information ####################################
# information : item에 대한 관점

def calcInfo(items_available, cur_theta):
    info = np.zeros(nitem)
    if model == 1:  # GRM
        for i in range(nitem):
            if items_available[i]:  # True
                ps = np.zeros(maxCat + 1)
                ps[0] = 1
                
                for k in range(maxCat - 1):
                    ps[k + 1] = 1 / (1 + np.exp(-1.7 * disc[i] * (cur_theta - beta[i, k])))

                prob = np.zeros(maxCat)                
                for k in range(maxCat):
                    prob[k] = ps[k] - ps[k + 1]
                    info[i] += (1.7 * disc[i] * (ps[k] * (1 - ps[k]) - ps[k + 1] * (1 - ps[k + 1]))) ** 2 / prob[k]

    elif model == 2:  # GPCM
        for i in range(nitem):
            if items_available[i]:
                zz = np.zeros(maxCat)
                sdsum = 0
                den = 0

                cb = np.zeros(int(maxCat))
                cb[1:] = beta[i]

                for k in range(maxCat):
                    sdsum += cb[k]
                    zz[k] = np.exp(1.7* disc[i] * (k * theta - sdsum))
                    den += zz[k]

                AX = 0
                BX = 0
                prob = np.zeros(maxCat)
                for k in range(maxCat):
                    prob[k] = zz[k] / den
                    AX += k ** 2 * prob[k]
                    BX += k * prob[k]

                info[i] = 1.7 ** 2 * disc[i] ** 2 * (AX - BX ^ 2)
    return info

############## Calculation likelihood weight information ################
def calc_LW_info(items_available, lk):
    info = np.zeros(nitem)
    info = np.sum(matrix_info * lk.reshape(-1, 1), axis=0)
    info[items_available == False] = 0
    return info


######## selection of maximum information ##############################
def select_maxInfo(ni_available, info_index, topN=1):  # topN : exposure control
    if ni_available >= topN:
        item_selected = info_index[np.random.choice(topN, 1)]

    else:
        item_selected = info_index[np.random.choice(ni_available, 1)]

    return item_selected

########## Calculation of SE ##############################################
def calcSE(examinee, ngiven, th, items_used):
    info = 0
    if model == 1:
        for i in range(ngiven):
            itm = items_used[examinee, i]
            ps = np.zeros(ncat[itm] + 1)
            ps[0] = 1
            ps[ncat[itm]] = 0
            for k in range(ncat[itm] - 1):
                ps[k + 1] = 1 / (1 + np.exp(-1.7 * disc[itm] * (th - beta[itm, k])))

            prob = np.zeros(ncat[itm])
            for k in range(ncat[itm]):
                prob[k] = ps[k]-ps[k+1]
                info += (1.7* disc[itm] * (ps[k] * (1-ps[k]) - ps[k + 1] * (1 - ps[k + 1]))) ** 2 / prob[k]

    elif model == 2:
        for i in range(ngiven):
            itm = items_used[examinee, i]

            cb = np.zeros(ncat[itm]+1)
            cb[1:] = beta[itm]

            zz = np.zeros(ncat[itm])
            sdsum = 0
            den = 0

            for k in range(ncat[itm]):
                sdsum += cb[k]
                zz[k] = np.exp(1.7 * disc[itm] * (k * th-sdsum))
                den += zz[k]

            AX = 0
            BX = 0
            prob = np.zeros(ncat[itm])
            for k in range(ncat[itm]):
                prob[k] = zz[k] / den
                AX += k ^ 2 * prob[k]
                BX += k * prob[k]

            info += 1.7 ** 2 * disc[itm] ** 2 * (AX - BX ** 2)

    SEM = 1 / np.sqrt(info)
    return SEM



########### Calculation EAP #############################################
'''
EAP : Expectation a posteriori / expected a posteriori
https://www.rasch.org/rmt/rmt163i.htm
'''
def calcEAP(examinee, ngiven, items_used, a):
    LH = np.ones(ntheta)
# ngiven이 0일때는 지나가는 for문: 
    for i in range(ngiven):
        item = int(items_used[examinee, i])
        resp = int(a[item])
        # resp = resp_data[examinee, item]
        prob = pp[resp-1, :, item] # pp.shape = (5, 81, 10)
        LH = LH * prob

    # 정규분포에 우도를 곱하기
    posterior = prior * LH
    EAP = np.sum(posterior * theta) / np.sum(posterior)

    if SE_method == 1:  # Posterior
        SEM = np.sqrt(np.sum(posterior * (theta - EAP) ** 2) / np.sum(posterior))

    elif SE_method == 2:  # Information
        SEM = calcSE(examinee, ngiven, EAP, items_used)

    # return(list(THETA=EAP,SEM=SEM,LH=LH,posterior=posterior))
    return EAP, SEM, LH, posterior  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior

########### Calculation MLE ############################################
def calcMLE(examinee, ngiven, items_used):
    '''
    MLE : 데이터가 필요.
    :param examinee:
    :param ngiven:
    :return:
    '''
    EAP_estimates = calcEAP(examinee, ngiven, items_used)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior

    total_raw = 0
    max_raw = 0

    ncat = np.zeros(ngiven)
    resp = np.zeros(ngiven)

    for i in range(ngiven):
        item = items_used[examinee, i]
        maxCat = item_par[item, -1]  # 모두 숫자이면 굳이 Pandas가 아니라 numpy를 이용하고 싶음
        resp[i] = resp_data[examinee, item]

    total_raw = np.sum(resp)
    max_raw = np.sum(ncat)
    if (total_raw == ngiven | total_raw == max.raw):
        MLE = EAP_estimates[0]
        SEM = EAP_estimates[1]
        # MLE = EAP_estimates["THETA"]
        # SEM = EAP_estimates["SEM"]
    else:
        maxIter = 50
        crit = 0.0001
        change = 1000
        nIter = 0
        post_theta = EAP_estimates[0]
        if model == 1: # GRM
            while nIter <= maxIter & change > crit:
                pre_theta = post_theta
                deriv1 = 0
                deriv2 = 0
                for i in range(ngiven):
                    item = items_used[examinee, i]
                    a = item_par.iloc[item, 0]
                    cb = item_par.iloc[item, 1:-1]

                    pp = np.zeros(maxCat)
                    ps = np.zeros(maxCat + 1)
                    qs = np.zeros(maxCat + 1)
                    ps[0] = 1
                    for k in range(maxCat - 1):
                        ps[k + 1] = 1 / (1 + np.exp(-1.7 * a * (pre_theta - cb[[k]])))
                        qs[k + 1] = 1 - ps[k + 1]

                    qs[maxCat + 1] = 1
                    pp[1] = 1 - ps[1]
                    pp[maxCat] = ps[maxCat]
                    for k in range(maxCat):
                        pp[k] = ps[k] - ps[k + 1]

                    deriv1 += 1.7 * a * (
                            (ps[resp[i]] * qs[resp[i]] - ps[resp[i] + 1] * qs[resp[i] + 1]) / pp[resp[i]])
                    deriv2 += 1.7 ** 2 * a ** 2 * ((ps[resp[i]] * qs[resp[i]] * (qs[resp[i]] - ps[resp[i]]) - ps[
                        resp[i] + 1] * qs[resp[i] + 1] * (qs[resp[i] + 1] - ps[resp[i] + 1])) / pp[resp[i]] - (
                                                        ps[resp[i]] * qs[resp[i]] - ps[resp[i] + 1] * qs[
                                                    resp[i] + 1]) ** 2 / pp[resp[i]] ^ 2)

                SEM = 1 / np.sqrt(abs(deriv2))
                post_theta = pre_theta - deriv1 / deriv2
                change = abs(post_theta - pre_theta)  # 변화
                nIter += 1

        elif model == 2: # GPCM
            while nIter <= maxIter & change > crit:
                pre_theta = post_theta
                deriv1 = 0
                deriv2 = 0
                for i in range(ngiven):
                    item = items_used[examinee, i]
                    # a<-item_par[item,"a"]
                    # cb<-unlist(item_par[item,paste("cb",1:(maxCat-1),sep="")]) # unlist : list를 vector로 변환
                    a = item_par.iloc[item, 0]
                    cb = item_par.iloc[item, 1:-1]

                    cb = cb(0, cb) # 원래 cb = c(0, cb) 이었는데 오타인것같아서 바꿈
                    pp = np.zeros(maxCat)
                    zz = np.zeros(maxCat)
                    sdsum = 0
                    den = 0

                    for k in range(maxCat):
                        sdsum += cb[k]
                        zz[k] = np.exp(1.7 * a * (k * pre_theta - sdsum))
                        den = den + zz[k]

                    AX = 0
                    BX = 0
                    for k in range(maxCat):
                        pp[k] = zz[k] / den
                        AX += k ^ 2 * pp[k]
                        BX += k * pp[k]

                    deriv1 += 1.7 * a * (resp[i] - BX)
                    deriv2 += 1.7 ^ 2 * a ^ 2 * (AX - BX ^ 2)

                deriv2 = - deriv2
                SEM = 1 / np.sqrt(abs(deriv2))
                post_theta = pre_theta - deriv1 / deriv2
                change = abs(post_theta - pre_theta)
                nIter += 1

        if post_theta < minTheta:
            MLE = minTheta
        elif post_theta > maxTheta:
            MLE = maxTheta
        else:
            MLE = post_theta

    # return값 = list(THETA=MLE,SEM=SEM,LH=EAP_estimates[LH],posterior=EAP_estimates[posterior])
    return MLE, SEM, EAP_estimates[2], EAP_estimates[posterior]

############## Plot Theta ###################################################
def plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data):
    '''
    :param j: 사람 번호
    :param ni_given: 사람 당 제공된 문제
    :param theta_history:
    :param estimates:
    :param items_used:
    :return:
    '''
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title("Estimation Audit Trail-Examinee " + str(j+1), fontsize=14)  # , position=(0.5, 1.0+0.05))
    x = range(1, ni_given + 1)
    plt.plot(x, theta_history[j, :ni_given], "-", color="royalblue")
    plt.axhline(y=estimates[0], ls='--', color='firebrick', linewidth=1)

    item_string = "items : " + ", ".join([str(int(i)) for i in items_used[j][:ni_given]])
    plt.text(ni_given / 2, maxTheta - 0.5, item_string, fontsize=10, ha="center")
    plt.text(ni_given / 2, minTheta + 0.4,
                "Theta: " + str(round(estimates[0], 2)) + ", SEM: " + str(round(estimates[1], 2)),
                fontsize=10, ha="center")
    plt.errorbar(x, theta_history[j, :ni_given], fmt='ro', yerr=1.96 * se_history[j, :ni_given], ecolor='black')
    plt.xticks(x)
    plt.yticks(range(minTheta, maxTheta + 1, 2))  # seq(minTheta,maxTheta,length=ni_given)
    plt.xlabel("Items Administered", fontsize=12)
    plt.ylabel("Theta", fontsize=12)

    ax2 = fig.add_subplot(1, 2, 2)
    plt.title("Final Posterior Distribution " + str(j+1), fontsize=14)
    plt.xlabel("Theta", fontsize=12)
    plt.ylabel("Posterior", fontsize=12)
    plt.plot(theta, estimates[3], "-", color="royalblue")

    resp_string = "Responses : " + ", ".join([str(int(a[int(i)])) for i in items_used[0][:ni_given]])

    plt.text(0, max(estimates[3]), resp_string, fontsize=10, ha="center")
    plt.xticks(range(minTheta, maxTheta + 1, 2))
    plt.show()


################### plot item usage ############################################
def plot_item_usage(items_used, theta_CAT):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)

    plt.plot(theta_CAT, np.sum(~np.isnan(items_used), axis=1), "o", color="royalblue")

    plt.xlabel("CAT Theta", fontsize=12)
    plt.ylabel("Number of Items Administered", fontsize=12)

    ax2 = fig.add_subplot(1, 2, 2)
    pct_items_used = np.zeros(nitem)

    tot_ni_used = np.sum(~np.isnan(items_used))

    for i in range(nitem):
        pct_items_used[i] = np.sum(items_used == i) * 100 / tot_ni_used

    plt.plot(range(1, nitem + 1), pct_items_used, "o", color="royalblue")
    plt.xticks(range(1, nitem + 1))
    plt.yticks(range(0, int(np.max(pct_items_used)) + 5, 5))
    for i in range(nitem):
        plt.vlines(i + 1, ymin=0, ymax=pct_items_used[i], color='red')
    plt.xlabel("Items", fontsize=12)
    plt.ylabel("How Many Used", fontsize=12)

    plt.show()

############### plot total item information ########################################
def plot_total_info():

    fig, host = plt.subplots()
    par1 = host.twinx()

    bank_info = np.array(matrix_info.sum(axis=1))
    bank_se = 1 / np.sqrt(bank_info)

    p1, = host.plot(theta, bank_info, "k-", label="Information")
    p2, = par1.plot(theta, bank_se, "k--", label="SEM")

    host.set_xlabel("Theta")
    host.set_ylabel("Total Information")
    par1.set_ylabel("Standard Error of Measurement")
    par1.set_ylim(0, 1)

    lines = [p1, p2]

    host.legend(lines, [l.get_label() for l in lines])
    plt.show()

############### plot item information ########################################
def plot_item_info():
    fig, ax = plt.subplots(int(np.ceil(nitem / 4)), 4, figsize=(15, 10))
    maxinfo = matrix_info.max()
    for i in range(nitem):
        x = i // 4
        y = i % 4
        ax[x, y].plot(theta, matrix_info[:, i], "-", color="black")
        max_theta = theta[np.argmax(matrix_info[:, i], axis=0)].round(1)
        ax[x, y].vlines(max_theta, ymin=0, ymax=maxinfo, color='red', linestyle='--')
        ax[x, y].set_title("Item " + str(i + 1))
        ax[x, y].set_yticks(range(0, int(np.ceil(maxinfo)) + 1))
        ax[x, y].set_xlabel("Theta")
        ax[x, y].set_ylabel("Information")
        ax[x, y].text(0, maxinfo - 0.15, " Max at Theta = " + str(max_theta), ha="center")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

################## plot item probability ####################################
def plot_item_prob():
    fig, ax = plt.subplots(int(np.ceil(nitem / 4)), 4, figsize=(15, 10))
    maxinfo = matrix_info.max()
    for i in range(nitem):
        x = i // 4
        y = i % 4
        style = ["k-", "r--", "g:", "r-.", "b-"]
        for j in range(5):
            ax[x, y].plot(theta, pp[j, :, i], style[j])
        ax[x, y].set_title("Item " + str(i + 1))
        ax[x, y].set_yticks(np.arange(0, 1.2, 0.2))
        ax[x, y].set_xlabel("Theta")
        ax[x, y].set_ylabel("Probability")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

def plot_total_ICC():
    fig, ax = plt.subplots(int(np.ceil(nitem / 4)), 4, figsize=(15, 10))
    for i in range(nitem):
        x = i // 4
        y = i % 4
        style = ["k-", "r--", "g:", "r-.", "b-"]
        for j in range(5):
            ax[x, y].plot(theta, pp[j, :, i], style[j])
'''
추후 사람에 관한 클래스 만들어야 함 -> db정보같은거

resp_data
theta_history
se_history 
이런거 담겨 있어야함

'''






########################################################################## 문제 제시 방식


if first_item_selection==2 and minTheta <= first_at_theta <= maxTheta:
    start_theta = first_at_theta
else:
    start_theta = prior_mean

pp, matrix_info = prep_prob_info()

plot_item_prob()
plot_item_info()
plot_total_info()



### 여기서부터 input문 수정
'''
if selection_method == 1:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(nitem, dtype=bool)
        ni_given = 0
        theta_current = start_theta

        while critMet == False and np.sum(items_available) > 0:
            
            array_info = calcInfo(items_available, theta_current)
            ni_available = np.sum(array_info > 0)  # count 함수와 동일

            info_index = array_info.argsort()[::-1]
            item_selected = select_maxInfo(ni_available, info_index)

            f = open('response.csv', 'w', newline='')
            wr = csv.writer(f)
            a[item_selected] = int(input("숫자 입력: "))
            wr.writerow([a])

            if ni_given == 0:
                if first_item_selection == 3 and 1 <= first_item <= ni:
                    if items_available[first_item]:
                        item_selected = first_item

            items_used[j, ni_given] = item_selected # array_info의 가장 큰 값 집어넣음 # information이 큰 문항을 사용한다고 생각하면 됨
            items_available[item_selected] = False # 사용해서 이제는 쓸 수 없는 문항은 false로!
            selected_item_resp[j, ni_given] = a[item_selected]  # 응답값 집어넣음. # numpy

            if interim_Theta == 1:  # EAP
                # j번째 사람의 ni_given(문항 몇번째까지 보고 있는지)에 따라 계산하는 것!으로 생각하면 됨s
                estimates = calcEAP(j, ni_given, items_used, a)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            # elif interim_Theta == 2:  # MLE
            # estimates = calcMLE(j, ni_given, items_used)  # THETA=MLE,SEM=SEM,LH=EAP_estimates[LH],posterior=EAP_estimates[posterior]
            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]

            if ni_given >= maxNI or (estimates[1] <= maxSE and ni_given >= minNI):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given # 총 몇개 문항을 administer했는가까지 보여줌

            ni_given += 1

        if (Show_Theta_Audit_Trail):
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, a)
            plot_item_usage(items_used, theta_CAT)
'''

'''
if selection_method == 1:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(nitem, dtype=bool)
        ni_given = 0
        theta_current = start_theta
        
        while critMet == False and np.sum(items_available) > 0:
            array_info = calcInfo(items_available, theta_current)
            ni_available = np.sum(array_info > 0)  # count 함수와 동일

            info_index = array_info.argsort()[::-1]
            item_selected = select_maxInfo(ni_available, info_index)

            if ni_given == 0:
                if first_item_selection == 3 and 1 <= first_item <= ni:
                    if items_available[first_item]:
                        item_selected = first_item

            items_used[j, ni_given] = item_selected # array_info의 가장 큰 값 집어넣음 # information이 큰 문항을 사용한다고 생각하면 됨
            items_available[item_selected] = False # 사용해서 이제는 쓸 수 없는 문항은 false로!
            #selected_item_resp[j][ni_given] = resp_data["R" + str(item_selected)][j] # 응답값 집어넣음. # pandas
            selected_item_resp[j, ni_given] = resp_data[j, item_selected]  # 응답값 집어넣음. # numpy

            if interim_Theta == 1:  # EAP
                # j번째 사람의 ni_given(문항 몇번째까지 보고 있는지)에 따라 계산하는 것!으로 생각하면 됨s
                estimates = calcEAP(j, ni_given, items_used, resp_data)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            # elif interim_Theta == 2:  # MLE
            # estimates = calcMLE(j, ni_given, items_used)  # THETA=MLE,SEM=SEM,LH=EAP_estimates[LH],posterior=EAP_estimates[posterior]
            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]

            if ni_given >= maxNI or (estimates[1] <= maxSE and ni_given >= minNI):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given # 총 몇개 문항을 administer했는가까지 보여줌

            ni_given += 1

            if (Show_Theta_Audit_Trail):
                plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
                plot_item_usage(items_used, theta_CAT)

elif selection_method == 2:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(ni, dtype=bool)
        # items_available[resp_data.iloc[j].isna()] = False #pandas
        items_available[np.isnan(resp_data[j, :])] = False  # numpy

        ni_given = 0
        theta_current = start_theta
        likelihood = np.ones(len(theta)) # selection_method 2에 추가

        while critMet == False and np.sum(items_available) > 0:
            array_info = calc_LW_info(items_available, likelihood)
            ni_available = np.sum(array_info > 0)

            info_index = array_info.argsort()[::-1]#오름차순정렬
            item_selected = select_maxInfo(ni_available, info_index)

            if ni_given == 0:
                if first_item_selection == 3 and 1 <= first_item <= ni:
                    if items_available[first_item]:
                        item_selected = first_item

            resp = resp_data[j, item_selected]-1
            prob = pp[resp, :, item_selected].reshape(-1) #prob = pp[:, item_selected_resp]
            likelihood *= prob
            items_used[j, ni_given] = item_selected
            items_available[item_selected] = False
            #selected_item_resp[j, ni_given] = resp_data[j, paste("R", item_selected, sep="")]

            if interim_Theta == 1:  # EAP
                estimates = calcEAP(j, ni_given, items_used, resp_data)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            elif interim_Theta == 2:  # MLE
                estimates = calcMLE(j, ni_given, items_used)  # THETA=MLE,SEM=SEM,LH=EAP_estimates[LH],posterior=EAP_estimates[posterior]

            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]


            if ni_given >= maxNI or (estimates[1] <= maxSE and ni_given >= minNI):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given
            ni_given += 1
   
        if (Show_Theta_Audit_Trail):     
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
    plot_item_usage(items_used, theta_CAT)

elif selection_method == 3:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(ni, dtype=bool)
        items_available[resp_data.iloc[j].isna()] = False
        ni_given = 0
        theta_current = start_theta
        posterior = prior
        while (critMet == False & sum(items_available) > 0):
            array_info = calc_PW_info(posterior)
            ni_available = np.sum(array_info > 0)
            info_index = array_info.argsort()[::-1]
            item_selected = select_maxInfo()
            if (ni_given == 0):
                if (first_item_selection == 3 & first_item >= 1 & first_item <= ni):
                    if (items_available[first_item] == True):
                        item_selected = first_item

            resp = resp_data[j, item_selected]
            prob = pp[:, item_selected, resp]
            posterior *= prob
            ni_given += 1
            items_used[j, ni_given] = item_selected
            items_available[item_selected] = False
            selected_item_resp[j, ni_given] < -resp.data[j, item_selected]

            estimates = calcEAP(j, ni_given)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]

            if (ni_given >= maxNI | (estimates[1] <= maxSE & ni_given >= minNI)):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given

        if (Show_Theta_Audit_Trail):
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
    plot_item_usage(items_used, theta_CAT)

elif selection_method == 4:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(ni, dtype=bool)
        items_available[resp_data.iloc[j].isna()] = False
        ni_given = 0
        theta_current = start_theta
        posterior = prior
        while (critMet == False & sum(items_available) > 0):
            array_info = calc_Expected_Info(posterior, theta_current)
            ni_available = np.sum(array_info > 0)
            info_index = array_info.argsort()[::-1]
            item_selected = select_maxInfo()
            if (ni_given == 0):
                if (first_item_selection == 3 & first_item >= 1 & first_item <= ni):
                    if (items_available[first_item] == True):
                        item_selected = first_item

            resp = resp_data[j, item_selected]
            prob = pp[:, item_selected, resp]
            posterior *= prob
            ni_given += 1
            items_used[j, ni_given] = item_selected
            items_available[item_selected] = False
            selected_item_resp[j, ni_given] = resp

            estimates = calcEAP(j, ni_given)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]
            if (ni_given >= maxNI | (estimates[1] <= maxSE & ni_given >= minNI)):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j,] = estimates[2]
                posterior_matrix[j,] = estimates[3]
                ni_administered[j] = ni_given

        if (Show_Theta_Audit_Trail):
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
    plot_item_usage(items_used, theta_CAT)

elif selection_method == 5:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(ni, dtype=bool)
        items_available[resp_data.iloc[j].isna()] = False
        ni_given = 0
        theta_current = start_theta
        posterior = prior

        while (critMet == False & np.sum(items_available) > 0):
            array_info = calc_Expected_Var(posterior, theta_current)
            ni_available = np.sum(array_info > 0)
            info_index = array_info.argsort()[::-1]
            item_selected = select_maxInfo()
            if (ni_given == 0):
                if (first_item_selection == 3 & first_item >= 1 & first_item <= ni):
                    if (items_available[first_item] == True):
                        item_selected = first_item

            resp = resp_data[j, item_selected]
            prob = pp[:, item_selected, resp]
            posterior *= prob
            ni_given += 1
            items_used[j, ni_given] = item_selected
            items_available[item_selected] = False
            selected_item_resp[j, ni_given] = resp

            estimates = calcEAP(j, ni_given)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]
            if (ni_given >= maxNI | (estimates[1] <= maxSE & ni_given >= minNI)):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given

        if (Show_Theta_Audit_Trail):
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
    plot_item_usage(items_used, theta_CAT)

elif selection_method == 6:
    for j in range(nExaminees):
        critMet = False
    items_available = np.ones(ni, dtype=bool)
    items_available[resp_data.iloc[j].isna()] = False
    ni_given = 0
    theta_current = start_theta
    posterior = prior
    while (critMet == False & sum(items_available) > 0):
        array_info = calc_Expected_PW_Info(posterior, theta_current)
        ni_available = np.sum(array_info > 0)
        info_index = array_info.argsort()[::-1]
        item_selected = select_maxInfo()
    
        if (ni_given == 0):
            if (first_item_selection == 3 & first_item >= 1 & first_item <= ni):
                if (items_available[first_item] == True):
                    item_selected = first_item
    
        resp = resp.data[j, item_selected]
        prob = pp[:, item_selected, resp]
    
        posterior *= prob
        ni_given += 1
        items_used[j, ni_given] = item_selected
        items_available[item_selected] = False
        selected_item_resp[j, ni_given] = resp
        estimates = calcEAP(j, ni_given)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
        theta_history[j, ni_given] = estimates[0]
        se_history[j, ni_given] = estimates[1]
        theta_current = estimates[0]
    
        if (ni_given >= maxNI | (estimates[1] <= maxSE & ni_given >= minNI)):
            critMet = True
            theta_CAT[j] = estimates[0]
            sem_CAT[j] = estimates[1]
            LH_matrix[j] = estimates[2]
            posterior_matrix[j] = estimates[3]
            ni_administered[j] = ni_given
    
    if (Show_Theta_Audit_Trail):
        plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
    plot_item_usage(items_used, theta_CAT)

elif selection_method == 7:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(ni, dtype=bool)
        items_available[resp_data.iloc[j].isna()] = False

        ni_given = 0
        random = np.random.uniform(size=ni)
        #is.na(random[!items_available]) = True
        # 사용 불가능한 것들만 랜덤으로 해서 ####원래코드 확인필요
        random[items_available] = True
        item_order = random.argsort()

        while (critMet == False & np.sum(items_available) > 0):
            item_selected = item_order[ni_given + 1]
            ni_given += 1
            items_used[j, ni_given] = item_selected
            items_available[item_selected] = False
            selected_item_resp[j, ni_given] = resp_data[j, item_selected]

            estimates = calcEAP(j, ni_given)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]
            if (ni_given >= maxNI | (estimates[1] <= maxSE & ni_given >= minNI)):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given

        if (Show_Theta_Audit_Trail):
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
    plot_item_usage(items_used, theta_CAT)

elif selection_method == 8:
    info_table = prep_info_ext_theta()
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(ni, dtype=bool)
        items_available[resp_data.iloc[j].isna()] = False
        ###
        info_table[j,not(items_available)]=0
        ni_given = 0
        # rev(order())  <=> array_info.argsort()[::-1]
        # <-rev(order(info_table[j,]))
        item_order = info_table[j].argsort()[::-1]

        while (critMet == False & np.sum(items_available) > 0):
            item_selected = item_order[ni_given + 1]
            ni_given += 1
            items_used[j, ni_given] = item_selected
            items_available[item_selected] = False
            selected_item_resp[j, ni_given] = resp_data[j, item_selected]
            estimates = calcEAP(j, ni_given)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior

            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]
            if (ni_given >= maxNI | (estimates[1] <= maxSE & ni_given >= minNI)):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given

        if (Show_Theta_Audit_Trail):
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)
    plot_item_usage(items_used, theta_CAT)

elif selection_method == 9:
    for j in range(nExaminees):
        critMet = False
        items_available = np.ones(ni, dtype=bool)
        items_available[resp_data.iloc[j].isna()] = False
        ni_given = 0
        theta_current = start_theta

        while (critMet == False & sum(items_available) > 0):
            array_info = calc_Loc_info(theta_current)
            ni_available = np.sum(array_info > 0)

            info_index = array_info.argsort()[::-1]
            item_selected = select_maxInfo()

            if (ni_given == 0):
                if (first_item_selection == 3 & first_item >= 1 & first_item <= ni):
                    if (items_available[first_item] == True):
                        item_selected = first_item

            ni_given += 1
            items_used[j, ni_given] = item_selected
            items_available[item_selected] = False
            selected_item_resp[j, ni_given] = resp_data[j, item_selected]

            estimates = calcEAP(j, ni_given)  # THETA=EAP,SEM=SEM,LH=LH,posterior=posterior
            theta_history[j, ni_given] = estimates[0]
            se_history[j, ni_given] = estimates[1]
            theta_current = estimates[0]
            if (ni_given >= maxNI | (estimates[1] <= maxSE & ni_given >= minNI)):
                critMet = True
                theta_CAT[j] = estimates[0]
                sem_CAT[j] = estimates[1]
                LH_matrix[j] = estimates[2]
                posterior_matrix[j] = estimates[3]
                ni_administered[j] = ni_given

        if (Show_Theta_Audit_Trail):
            plot_theta_audit_trail(j, ni_given, theta_history, se_history, estimates, items_used, resp_data)

    plot_item_usage(items_used, theta_CAT)

'''