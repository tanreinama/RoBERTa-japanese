# RoBERTa-japanese

Japanese BERT Pretrained Model

RoBERTaとは、[Liu, Yinhanらが提案](https://arxiv.org/abs/1907.11692)する、BERTの改良版です。モデル構造そのものはオリジナルのBERTと同じで、学習手法に工夫があります。

このプロジェクトは、[スポンサーを募集しています](report/sponsor.md)。



# RoBERTa (改良BERT)  日本語モデル

***<font color='red'>New</font>***

- [GitHub Sponserによるスポンサーシップを開始しました](https://github.com/sponsors/tanreinama)

- [事前学習済みbaseモデルを公開しました](report/models.md)

  


## RoBERTa (改良BERT) の日本語版です

### 学習済みモデルについて

[report/models.md](report/models.md)

### 学習させたコーパスについて

[report/corpus.md](https://github.com/tanreinama/gpt2-japanese/blob/master/report/corpus.md)

### スポンサーシップについて

[report/sponsor.md](report/sponsor.md)



### TODO

✓smallモデルの公開（2020/12/6）<br>✓baseモデルの公開（2021/1/4）<br>



# 使い方



GitHubからコードをクローンします

```sh
$ git clone https://github.com/tanreinama/RoBERTa-japanese
$ cd RoBERTa-japanese
```

モデルファイルをダウンロードして展開します

```sh
$ wget https://www.nama.ne.jp/models/RoBERTa-ja_base.tar.bz2
$ tar xvfj RoBERTa-ja_base.tar.bz2
```

分かち書きに使う[Japanese-BPEEncoder](https://github.com/tanreinama/Japanese-BPEEncoder)をコピーします

```sh
$ git clone https://github.com/tanreinama/Japanese-BPEEncoder.git
$ cp Japanese-BPEEncoder/* ./
```



##  クラス分類サンプル



クラス分類問題は、BERTモデルに入力される「[CLS]」トークンに対するtransformerの出力を、全結合層で分類します。

サンプルプログラムは、

```
dir/<classA>/textA.txt
dir/<classA>/textB.txt
dir/<classB>/textC.txt
・・・
```

のように、「クラス名/ファイル」という形でテキストファイルが保存されている前提で、テキストファイルをクラス毎に分類するモデルを学習します。

ここでは、[livedoor ニュースコーパス](http://www.rondhuit.com/download.html#ldcc)を使用する例をサンプルとして提示します。

まず、コーパスをダウンロードして展開すると、「text」以下に記事の入っているディレクトリが作成されます。

```sh
$ wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
$ tar xvfz ldcc-20140209.tar.gz
$ ls text/
```

### クラス分類モデルの学習

ここではファイル1つを1データとして扱うので、「--input」にコーパスのディレクトリを、「--model」にモデル名を入れて、「train-classifier.py」を実行します。

学習したモデルは、「checkpoint」以下の「--run_name」で指定したディレクトリ内に保存されます。

```sh
$ python train-classifier.py --input text --model RoBERTa-ja_base --run_name run_classifier1
```

以下のようにサブディレクトリ名とクラスIDとの対応が表示された後、学習が進みます。

```sh
text/dokujo-tsushin mapped for id_0, read 871 contexts.
text/it-life-hack mapped for id_1, read 871 contexts.
text/kaden-channel mapped for id_2, read 865 contexts.
text/livedoor-homme mapped for id_3, read 512 contexts.
text/movie-enter mapped for id_4, read 871 contexts.
text/peachy mapped for id_5, read 843 contexts.
text/smax mapped for id_6, read 871 contexts.
text/sports-watch mapped for id_7, read 901 contexts.
text/topic-news mapped for id_8, read 771 contexts.
```

ここでは1ファイルが1データとして扱っていますが、「--train_by_line」オプションを指定することで、テキストファイル内のテキスト1行ずつを1データとして扱うことも出来ます。

### クラス分類の推論

推論は、「predict-classifier.py」を実行します。入力テキストの形式は学習時と同じです。

```sh
$ python predict-classifier.py --input text --model checkpoint/run_classifier1
```

出力先は、分類結果のcsvファイルで、デフォルトは「predict.csv」です。

「--output_file」オプションで変更出来ます。

```sh
$ head -n5 predict.csv
id,y_true,y_pred
dokujo-tsushin/dokujo-tsushin-6140446.txt,0,0
dokujo-tsushin/dokujo-tsushin-6502782.txt,0,0
dokujo-tsushin/dokujo-tsushin-5292054.txt,0,0
dokujo-tsushin/dokujo-tsushin-6340031.txt,0,0
```

分類結果を、タイル表示で可視化した結果も、「出力ファイル名_map.png」という名前で保存されます。

![predict_map.png](predict_map.png)



## 文章のベクトル化



RoBERTaのモデルはtransformerベースなので、入力されたBPEに対して、それぞれの対応するベクトルを出力します。

### transform

入力文章に対して、分かち書きしたBPEに対応するベクトルを全て出力するには、「bert-transform.py」を起動します。

すると、「--context」で指定した文章を分かち書きして、対応するBPEのベクトルの列を返します。

```sh
$ python bert-transform.py --context "俺の名前は坂本俊之。何処にでもいるサラリーマンだ。" --model RoBERTa-ja_base
input#0:
[[-0.03498905 -0.39651284  0.08991005 ... -0.22224797 -0.47363394
   0.24207689]
 [-1.5242212  -0.376556    0.00792967 ...  0.79690623 -0.10436316
  -0.8417113 ]
 [ 0.32648042  0.63716465  0.53898436 ... -0.43589458 -1.5498768
  -1.29899   ]
・・・（略）
```

### 文章全体のベクトル化

BPEではなく、文章に対応するベクトルは、「[CLS]」トークンに対する出力ベクトルを利用します。

文章のベクトル化を行うには、「bert-transform.py」を「--output_cls」オプションを指定して起動します。

```sh
$ python bert-transform.py --context "俺の名前は坂本俊之。何処にでもいるサラリーマンだ。" --model RoBERTa-ja_base --output_cls
input#0:
[-2.73364842e-01  2.00292692e-01  2.09339023e-01 -4.24192771e-02
 -1.32657290e-01  9.70005244e-03 -6.21969163e-01  1.17966272e-02
 -2.89265156e-01 -4.36325669e-02  5.75233221e-01  7.28546530e-02
 -1.82987601e-01 -3.07042986e-01  1.49002954e-01 -2.09294081e-01
・・・（略）
```

すると、文章を固定長のベクトルに出来ます。



## テキストの穴埋め



RoBERTaのモデルは入力されたテキスト内の「[MASK]」部分を予測します。

「bert-predict.py」で、直接穴埋め問題を解かせることが出来ます。

「[MASK]」一つでエンコード後のBPE一つなので、「[MASK]」が日本語2文字か1文字になります。

```sh
$ python bert-predict.py --context "俺の名前は坂本[MASK]。何処にでもいるサラリー[MASK]だ。" --model RoBERTa-ja_base
俺の名前は坂本龍馬。何処にでもいるサラリーマンだ。
```



##  モデルのファインチューニング



[コーパス2020](https://github.com/tanreinama/gpt2-japanese/blob/master/report/corpus.md)でプレトレーニングしたモデルは公開しています。ここでの手順は、独自のデータでモデルをさらにファインチューニングする方法です。

### エンコード

[Japanese-BPEEncoder](https://github.com/tanreinama/Japanese-BPEEncoder)を使用して、学習させたい独自のデータをエンコードします。

```sh
$ git clone https://github.com/tanreinama/Japanese-BPEEncoder.git
$ cd Japanese-BPEEncoder
$ python encode_bpe.py --src_dir <content file path> --dst_file finetune
$ mv finetune.npz ../
$ cd ..
```

### 学習

「--base_model」に元のプレトレーニング済みモデルを「--dataset 」にエンコードしたファイルを指定して、「run_finetune.py」を起動します。

```sh
$ python run_finetune.py --base_model RoBERTa-ja_base --dataset finetune.npz --run_name RoBERTa-finetune_run1
```

学習したモデルは、「checkpoint」以下の「--run_name」で指定したディレクトリ内に保存されます。



# REFERENCE

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
