# SAM_mask_tool
SAM(https://segment-anything.com/)を使ってマスク画像を作成するツールです。

# 事前準備
- https://github.com/facebookresearch/segment-anything からViT-H SAM modelをダウンロードしてフォルダ直下に保存
- segment-anythingの実行環境を構築
- gui_auto.pyファイルのimpathに入力画像へのパスを設定してください。
- gui_auto.pyファイルのdeviceに使用するデバイスをしてしてください。（GPUがなければ"cpu"に変更）

# 使い方
下記のコマンドで実行できます。
```bash
python gui_auto.py
```

画面が3つ表示されます。
- 元画像
- SAMセグメンテーション画像
- マスク画像

すべての画面に共通の下記のコマンドを使って、マスク画像を作成します。
- 左クリック：クリックした領域をマスクに追加
- 右クリック：クリックした領域をマスクから消す
- sキー：マスク画像をmaskフォルダに保存
- rキー：マスク画像を初期化
- qキー：終了
- 1キー：領域選択モード
- 2キー：塗り絵モード
