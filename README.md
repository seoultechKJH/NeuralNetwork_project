# NeuralNetwork_project
재무제표 정보를 활용해 기업의 업종을 분류하는 Feedforward Neural Network 모델입니다.

# Dataset
dataset은 전자공시시스템에서 제공되는 2015, 2016, 2018년도의 재무상태표 종합 파일입니다.
기존 dataset에는 산업표준분류에 따라 18개 업종이 부여되어 있지만, 일부 소수 클래스를 제거하고 4개의 업종 카테고리로 통합했습니다. (0-유통업, 1-서비스업, 2-제조업, 3-전기건설업)

#Preprocessing
scaling - min_max_scaler 사용
oversampling - SMOTE 사용
train_test_split : train_set (80%), test_set (20%)

#hyper parameter
input = 9 (재무지표 개수와 동일)
number of layer = 2
activation function = ReLU
output = 4 (업종 수와 동일)
epochs = 1000
optimization function = Rprop
learning rate = 0.001

#result
제조업에 대한 분류 성능은 월등히 좋음

#limitation
손익계산서 정보가 누락되었기 때문에 다양한 재무지표를 수집되지 못했으며, 손익계산서로부터 파악 가능한 매출채권, 매입채무, 영업이익 등의 정보가 수집된다면 다른 업종의 분류 성능도 좋아질 것으로 예상됩니다.
