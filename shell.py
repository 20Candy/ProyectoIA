import tkinter as tk
import pickle

model1 = pickle.load(open('best_lr1.sav', 'rb'))
vectorizer1 = pickle.load(open('vectorizer1.sav', 'rb'))

model2 = pickle.load(open('best_lr2.sav', 'rb'))
vectorizer2 = pickle.load(open('vectorizer2.sav', 'rb'))

class ShellSimulator(tk.Frame):
    def __init__(this, master=None):
        super().__init__(master)
        this.master = master
        this.create_widgets()

        this.start = 0
        this.end = 0
        this.charBuffer = 0
        this.userInput = ""
        this.commandHistory = []
        this.historyPointer = -1
    
    def keyPress(this, event):
        if event.keysym == "Return":
            this.charBuffer = 0

            if (len(this.userInput) > 0 and 
                len(this.commandHistory) == 0) or (
                len(this.userInput) > 0 and
                len(this.commandHistory) > 0 and
                not this.userInput == this.commandHistory[-1]
                ):
                this.commandHistory.append(this.userInput)

            this.execute_command(this.userInput)
            this.historyPointer = -1
            this.userInput = ""
            return "break"
            
        elif event.keysym == "BackSpace":
            if this.charBuffer == 0:
                return "break"
            else:
                this.charBuffer -= 1
                this.userInput = this.userInput[:-1]
        
        elif event.keysym == "Up":
            if this.historyPointer < len(this.commandHistory) - 1:
                this.historyPointer += 1
                this.userInput = this.commandHistory[this.historyPointer]
                this.textbox.delete("end linestart -1 lines", tk.END)
                this.textbox.insert(tk.END, "\nCyberBullying Detector> ")
                this.textbox.insert(tk.END, this.userInput)

            return "break"
        
        elif event.keysym == "Down":
            changePointer = False
            if this.historyPointer >= 1:
                this.historyPointer -= 1
                this.userInput = this.commandHistory[this.historyPointer]
                this.textbox.delete("end linestart -1 lines", tk.END)
                this.textbox.insert(tk.END, "\nCyberBullying Detector> ")
                this.textbox.insert(tk.END, this.userInput)
            elif this.historyPointer == 0:
                changePointer = True
                this.userInput = ""
                this.textbox.delete("end linestart -1 lines", tk.END)
                this.textbox.insert(tk.END, "\nCyberBullying Detector> ")
                this.textbox.insert(tk.END, this.userInput)

            if changePointer:
                this.historyPointer = -1
                
            return "break"
        
        elif event.keysym == "Left":
            if this.charBuffer == 0:
                return "break"
            else:
                this.charBuffer -= 1

        elif event.keysym == "Right":
            if this.charBuffer == 0 or this.charBuffer >= len(this.userInput):
                return "break"
            else:
                this.charBuffer += 1
        
        else:
            this.charBuffer += 1
            this.userInput += event.char

    def create_widgets(this):
        this.textbox = tk.Text(this.master, bg='black', fg='white')
        this.textbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        this.textbox.bind('<Key>', this.keyPress)
        this.textbox.config(insertbackground='white') # Set the cursor color to white
        this.textbox.focus_set()

    def execute_command(this, userInput):
        print("\nuserInput: " + userInput+"\n")
        running = True

        # EXIT ===================================================================================================================
        if userInput == "exit":
            this.master.destroy()
            running = False

        else:
            newTweet = vectorizer1.transform([userInput]).toarray()
            result = model1.predict(newTweet)

            if result[0] == "cyberbullying":
                this.textbox.insert(tk.END, "\nEl tweet es cyberbullying")
                newTweet = vectorizer2.transform([userInput]).toarray()
                result = model2.predict(newTweet)

                resultString = "Tipo de cyberbullying: " + result[0] + "\n"
                this.textbox.insert(tk.END, "\n" + resultString)
            else:
                this.textbox.insert(tk.END, "\nEl tweet no es cyberbullying\n")
      


        if (running):
            this.textbox.insert(tk.END, "\nCyberBullying Detector> ")
            this.textbox.mark_set(tk.INSERT, "end-1c") # Move the cursor to the end of the textbox


root = tk.Tk()
root.configure(bg='black')
root.title("HBase Simulator")
app = ShellSimulator(master=root)
app.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
app.textbox.insert(tk.END, "CyberBullying Detector> ")
app.textbox.mark_set(tk.INSERT, "end-1c") # Move the cursor to the end of the textbox
root.mainloop()