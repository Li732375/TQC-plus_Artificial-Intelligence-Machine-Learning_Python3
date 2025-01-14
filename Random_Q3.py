import random
import os
import pickle

class QuestionPicker:
    def __init__(self):
        # 題目類別
        self.category1 = [101, 102, 103, 104, 105]
        self.category2 = [201, 202, 203, 204, 205]
        self.category3 = [302, 303, 304, 305]
        
        # 儲存已抽過的題目
        self.state_file = 'picker_state.pkl'
        
        # 嘗試讀取之前的抽選狀態
        self.load_state()

    def load_state(self):
        # 如果檔案存在，讀取之前的狀態
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as file:
                state = pickle.load(file)
                self.remaining1 = state.get('remaining1', self.category1.copy())
                self.remaining2 = state.get('remaining2', self.category2.copy())
                self.remaining3 = state.get('remaining3', self.category3.copy())
        else:
            # 如果檔案不存在，初始化
            self.remaining1 = self.category1.copy()
            self.remaining2 = self.category2.copy()
            self.remaining3 = self.category3.copy()

    def save_state(self):
        # 將當前狀態儲存到檔案
        state = {
            'remaining1': self.remaining1,
            'remaining2': self.remaining2,
            'remaining3': self.remaining3
        }
        with open(self.state_file, 'wb') as file:
            pickle.dump(state, file)

    def reset_category(self, category_number):
        # 重置指定類別的題目
        if category_number == 1:
            self.remaining1 = self.category1.copy()
        elif category_number == 2:
            self.remaining2 = self.category2.copy()
        elif category_number == 3:
            self.remaining3 = self.category3.copy()

    def pick_question(self):
        # 如果某一類的題目數量不足，觸發重置機制
        if not self.remaining1:
            #print("一類題目不足，重置一類題目！")
            self.reset_category(1)
        
        if not self.remaining2:
            #print("二類題目不足，重置二類題目！")
            self.reset_category(2)
        
        if not self.remaining3:
            #print("三類題目不足，重置三類題目！")
            self.reset_category(3)
        
        # 抽選題目
        question1 = random.choice(self.remaining1)
        question2 = random.choice(self.remaining2)
        question3 = random.choice(self.remaining3)

        # 移除已抽過的題目
        self.remaining1.remove(question1)
        self.remaining2.remove(question2)
        self.remaining3.remove(question3)

        # 保存當前狀態
        self.save_state()

        return question1, question2, question3

# 使用範例
picker = QuestionPicker()

# 模擬多次抽選
q1, q2, q3 = picker.pick_question()
print(f"題目: {q1}, {q2}, {q3}")
