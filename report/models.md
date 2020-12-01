# RoBERTa-japanese pretrain models


RoBERTa-japanese Pretrained Model



## 学習済みRoBERTa (改良BERT) 日本語モデル



一旦、モデルの公開は停止しています



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

現在、baseモデルは公開中、largeモデルは学習中です。largeモデルも学習が終わり次第公開する予定です。



| モデル | レイヤー数       | ダウンロードURL                                              |
| ------ | ---------------- | ------------------------------------------------------------ |
| base   | 12heads,12layers | https://www.nama.ne.jp/models/RoBERTa-ja_base.tar.bz2<br />（予備URL：http://ailab.nama.ne.jp/models/RoBERTa-ja_base.tar.bz2） |

