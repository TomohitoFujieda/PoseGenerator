import launch

# 日本語言語分析用ライブラリのインストール
if not launch.is_installed("fugashi"):
    launch.run_pip("install fugashi", "(fugashi) NPL for Japanese")
if not launch.is_installed("ipadic"):
    launch.run_pip("install ipadic", "(ipadic) NPL for Japanese")
# 出力結果ダウンロード用ライブラリのインストール
# if not launch.is_installed():
#     launch.run_pip("install tkinter", "(tkinter) for save files")