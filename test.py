#!/usr/bin/env python3
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # データセットの生成
    tx=[[1.1,2.2,3.0,4.0],[2.0,3.0,4.0,1.0],[2.0,2.0,3.0,4.0]]
    tx=np.asarray(tx,dtype=np.float32)
    tt=[0,1,2]
    tt=tf.convert_to_tensor(tt)

    # ネットワークの定義
    model=Network()
    cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
    acc=tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
    
    # 学習1回の記述
    @tf.function
    def inference(tx,tt):
        with tf.GradientTape() as tape:
            ty=model.call(tx)
            costvalue=cce(tt,ty) #正解と出力の順番はこの通りにする必要がある．
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    # 学習ループ
    for epoch in range(1,3000+1): # 学習の回数の上限値
        traincost,trainacc=inference(tx,tt)
        if epoch%100==0:
            print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch,traincost,trainacc))
    
    # 学習が本当にうまくいったのか入力ベクトルのひとつを処理させてみる
    tx1=np.asarray([[1.1,2.2,3.0,4.0]],dtype=np.float32)
    ty1=model.call(tx1)
    print(ty1)

    # 未知のデータを読ませてみる
    tu=np.asarray([[999,888,777,666]],dtype=np.float32)
    tp=model.call(tu)
    print(tp)

    # Denseの最初の引数の値やエポックの値や変化させて，何が起こっているか把握する．

class Network(Model):
    def __init__(self):
        super(Network,self).__init__()
        self.d1=Dense(10, activation="relu") # これは全結合層を生成するための記述．
        self.d2=Dense(3, activation="softmax")
    
    def call(self,x):
        y=self.d1(x)
        y=self.d2(y)
        return y

if __name__ == "__main__":
    main()