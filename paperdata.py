# 送线
import math
weigh_xianquan_1 = 229.0
every_layer_daoxian_2 = 88.0
weigh_daoxian_3 = 2.4

weigh_zhidai_5 = 15.0
high_zhidai_6 = 2.32
min_Gradient_insulation_7 = 2
max_Gradient_insulation_7 = 4
thickness_zhidai_8 = 0.1



song_xian = (weigh_xianquan_1 - weigh_zhidai_5 * 2 - weigh_daoxian_3) / every_layer_daoxian_2
print(song_xian)

duan_jueyuan_zhidai = math.ceil((high_zhidai_6 / thickness_zhidai_8) * 2)
print(duan_jueyuan_zhidai)

Insulation_layer_turns = math.ceil((weigh_xianquan_1 - weigh_zhidai_5 * 2) / weigh_zhidai_5) * (min_Gradient_insulation_7 + max_Gradient_insulation_7) /2 + 1
print(Insulation_layer_turns)

# self.pushButton.clicked.connect(self.get_songxian)
# self.pushButton_2.clicked.connect(self.get_Endpapertape)
# self.pushButton_3.clicked.connect(self.get_Insulation_layer_turns)
#     def get_songxian(self):
#         self.textEdit.clear()
#         weigh_xianquan_1 = float(self.lineEdit.text())
#         weigh_zhidai_5 = float(self.lineEdit_5.text())
#         weigh_daoxian_3 = float(self.lineEdit_3.text())
#         every_layer_daoxian_2 = int(self.lineEdit_2.text())
#         if weigh_xianquan_1 != None:
#             # song_xian = (weigh_xianquan_1 - weigh_zhidai_5 * 2 - weigh_daoxian_3) / every_layer_daoxian_2
#             # self.textEdit.append(str(song_xian)[:5])
#             print('yes')
#         else:
#             print("none!")
#
#     def get_Endpapertape(self):
#         self.textEdit_2.clear()
#         high_zhidai_6 = float(self.lineEdit_6.text())
#         thickness_zhidai_8 = float(self.lineEdit_9.text())
#         duan_jueyuan_zhidai = math.ceil((high_zhidai_6 / thickness_zhidai_8) * 2)
#         self.textEdit_2.append(str(int(duan_jueyuan_zhidai)))
#
#     def get_Insulation_layer_turns(self):
#         self.textEdit_3.clear()
#         weigh_xianquan_1 = float(self.lineEdit.text())
#         weigh_zhidai_5 = float(self.lineEdit_5.text())
#         min_Gradient_insulation_7 = int(self.lineEdit_8.text())
#         max_Gradient_insulation_7 = int(self.lineEdit_7.text())
#         Insulation_layer_turns = math.ceil((weigh_xianquan_1 - weigh_zhidai_5 * 2) / weigh_zhidai_5) * (
#                         min_Gradient_insulation_7 + max_Gradient_insulation_7) / 2 + 1
#         self.textEdit_3.append(str(int(Insulation_layer_turns)))
#
#
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())