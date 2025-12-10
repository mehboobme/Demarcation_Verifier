"""Test GUI checkbox state"""
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Checkbox Test")

use_ai = tk.BooleanVar(value=True)

check = ttk.Checkbutton(
    root,
    text="Enable GPU Typology Verifier",
    variable=use_ai
)
check.pack(pady=20)

def show_value():
    value = use_ai.get()
    result_label.config(text=f"Checkbox value: {value}\nUse GPU: {value}")
    print(f"Checkbox state: {value}")

btn = tk.Button(root, text="Check Value", command=show_value)
btn.pack(pady=10)

result_label = tk.Label(root, text="Click button to check value")
result_label.pack(pady=10)

root.mainloop()
