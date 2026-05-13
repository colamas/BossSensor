# -*- coding: utf-8 -*-
import tkinter as tk
from PIL import Image, ImageTk


def show_image(image_path='s_pycharm.jpg'):
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.bind('<Escape>', lambda e: root.destroy())

    img = Image.open(image_path)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    img = img.resize((screen_w, screen_h), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=photo, bg='black')
    label.pack(expand=True, fill='both')
    root.mainloop()


if __name__ == '__main__':
    show_image()
