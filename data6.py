import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
le.transform(['M', 'B'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=1)

# 連結する処理としてスケーリング、主成分分析、ロジスティク回帰を指定
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# 分割元データ、分割数、乱数生成器の状態を指定し、
# 層化k分割交差検証イテレーターを表すStratifiedKFoldクラスのインスタンス化
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
# イテレーターのインデックスと要素をループ処理（上から順に）
# データをモデルに適合
# テストデータの正解率を算出
# リストに正解率を追加
# 分割の番号、0以上の要素数、正解率を出力
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s,Acc:%.3f' %
          (k+1, np.bincount(y_train[train]), score))
# 正解率の平均と標準差を出力
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# 交差検証のcross_val_score関数でモデルの正解率を算出
# 推定器estimator、訓練データX、予測値ｙ、分割数cv、CPU数n_jobsを指定
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train, y=y_train,
                         cv=10, n_jobs=1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
