import numpy as np

class Perceptron:
    """
    퍼셉트론

    매개변수
    ______________
    eta : float
        학습률 (0.1 ~ 1.0)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    ______________
    w_ : 1d-array
        학습된 가중치
    b_ : 스칼라
        학습된 절편 유닛

    errors_ : list
        에포크마다 누적된 분류 오류
    """

    def __init__(self,eta = 0.01, n_iter = 50, random_sate = 1 ):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_sate

    def fit(self,X,y):
        """
        훈련 데이터 학습

        매개변수
        _________
        X : {array-like}, shape = [n_samples, n_features]
            n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]
            타깃 값

        반환값
        _________
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        """입력 계산"""
        return np.dot(X,self.w_) + self.b_

    def predict(self,X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환 합니다"""
        return np.where(self.net_input(X) >= 0.0 , 1, 0)