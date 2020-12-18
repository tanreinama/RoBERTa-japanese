# RoBERTa-japanese pretrain models


RoBERTa-japanese Pretrained Model



## 学習済みRoBERTa (改良BERT) 日本語モデル



Smallモデル→→→[ダウンロード](https://www.nama.ne.jp/models/RoBERTa-ja_small.tar.bz2) （[予備URL](http://ailab.nama.ne.jp/models/RoBERTa-ja_small.tar.bz2)）←←←



## RoBERTa (改良BERT) とは



[Liu, Yinhanらが提案](https://arxiv.org/abs/1907.11692)する、BERTの改良版です。

RoBERTaの、BERTからの改良点は学習手法の改良のみで、モデル構造そのものはオリジナルのBERTそのものです（[こちら](https://ai-scholar.tech/articles/others/roberta-ai-230)の記事などが詳しいです）。



## RoBERTa-japaneseとは



日本語コーパス（[コーパス2020](https://github.com/tanreinama/gpt2-japanese/blob/master/report/corpus.md)）を学習させたモデルです。

オリジナルのBERTに、RoBERTaの論文から、以下のFEATUREsを導入して作成しました。

- dynamic mask
- NSPは使わない
- FULL-SENTENCESな学習
- バッチサイズとlr値を最適化

分かち書き/エンコードは[Japanese-BPEEncoder](https://github.com/tanreinama/Japanese-BPEEncoder)を使用します。そのため、オリジナルのRoBERTaからも、語彙数について違いがあります。また、分かち書きがBPEエンコードで、単語単位ではないので、[MASK]もBPE単位になっています。



## 公開している学習済みモデル



RoBERTaの論文と同じく、baseとlargeの二種類のモデルがあります。レイヤー数はオリジナルと同じですが、語彙数が異なるため、出力層のlogitsのパラメーター数が異なり、総パラメーター数もオリジナルと異なっています。

現在、smallモデルは公開中、baseモデルは学習中です。baseモデルも学習が終わり次第公開する予定です。



| モデル | 隠れ層次元 | レイヤー数     | 学習バッチサイズ | 学習回数 | ダウンロードURL                                              |
| ------ | ---------- | -------------- | ---------------- | -------- | ------------------------------------------------------------ |
| small  | 512        | 4heads,4layers | 16K              | 330K     | https://www.nama.ne.jp/models/RoBERTa-ja_small.tar.bz2<br />（予備URL：http://ailab.nama.ne.jp/models/RoBERTa-ja_small.tar.bz2） |

