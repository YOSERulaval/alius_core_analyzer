import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

from machine_learning_model.useModel import ModelPredictor


class ModelTrainerFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        # Variables globales
        self.train_path = ""
        self.train_file_name = os.path.basename(self.train_path)

        self.val_path = ""
        self.val_file_name = os.path.basename(self.val_path)

        self.grid_columnconfigure(0, weight=1)

        # Titre du frame
        self.title = ctk.CTkLabel(self, text="Model Trainer", fg_color="white", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=2)

        # Boutons d'import/export des dataset
        # Importer un fichier json avec les annotations pour entraînement
        self.import_train_button = ctk.CTkButton(self, text="Import training data", command=self.train_path_picker)
        self.import_train_button.grid(column=0, row=1, padx=10, pady=10, sticky="w", columnspan=1)

        # Importer un fichier json avec les annotations pour test
        self.import_test_button = ctk.CTkButton(self, text="Import test data", command=self.val_path_picker)
        self.import_test_button.grid(column=0, row=2, padx=10, pady=10, sticky="w", columnspan=1)

        # Afficheur de nom des fichiers sélectionnés (pour train et test) # TODO
        self.train_path_label = ctk.CTkLabel(self, text=self.train_file_name, fg_color="white", corner_radius=6)
        self.train_path_label.grid(column=1, row=1, padx=10, pady=10, sticky="ew", columnspan=2)

        self.val_path_label = ctk.CTkLabel(self, text=self.val_file_name, fg_color="white", corner_radius=6)
        self.val_path_label.grid(column=1, row=2, padx=10, pady=10, sticky="ew", columnspan=2)

        # Gestion des paramètres :
        # Nombre d'itérations
        self.param1 = ctk.CTkLabel(self, text="Nombre d'itération", fg_color="white", corner_radius=6)
        self.param1.grid(row=3, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param1_option = ctk.CTkOptionMenu(self, values=["300", "500", "800", "1000"], command=self.optionmenu_callback)
        self.param1_option.grid(row=3, column=1, padx=10, pady=10, sticky="w", columnspan=1)

        # Nombre de classe
        self.param2 = ctk.CTkLabel(self, text="Nombre de classe", fg_color="white", corner_radius=6)
        self.param2.grid(row=4, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param2_option = ctk.CTkOptionMenu(self, values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                                               command=self.optionmenu_callback)
        self.param2_option.grid(row=4, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Batch size par image "RoIHead batch size"
        self.param3 = ctk.CTkLabel(self, text="RoIHead batch size", fg_color="white", corner_radius=6)
        self.param3.grid(row=5, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param3_option = ctk.CTkOptionMenu(self, values=["128", "256", "384", "512", "640", "768", "896"], command=self.optionmenu_callback)
        self.param3_option.set("512")
        self.param3_option.grid(row=5, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # real batch size
        self.param4 = ctk.CTkLabel(self, text="Real batch size", fg_color="white", corner_radius=6)
        self.param4.grid(row=6, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param4_option = ctk.CTkOptionMenu(self, values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], command=self.optionmenu_callback)
        self.param4_option.set("2")
        self.param4_option.grid(row=6, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # learning rate
        self.param5 = ctk.CTkLabel(self, text="Learning rate LR", fg_color="white", corner_radius=6)
        self.param5.grid(row=7, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param5_option = ctk.CTkOptionMenu(self, values=
        ["0.00020", "0.00025", "0.00030", "0.00035", "0.00040", "0.00045", "0.00050", "0.00055", "0.00060", "0.00065"],
                                               command=self.optionmenu_callback)
        self.param5_option.set("0.00025")
        self.param5_option.grid(row=7, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # model device
        self.param4 = ctk.CTkLabel(self, text="Model device", fg_color="white", corner_radius=6)
        self.param4.grid(row=8, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param4_option = ctk.CTkOptionMenu(self, values=["cpu", "cuda"], command=self.optionmenu_callback)
        self.param4_option.grid(row=8, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Begin training button
        self.train_button = ctk.CTkButton(self, text="Train !", corner_radius=6, command=self.train_button_callback)
        self.train_button.grid(row=9, column=0, padx=10, pady=10, sticky="ew", columnspan=2)


    def train_path_picker(self):
        root = tk.Tk()
        root.withdraw()

        self.train_path = filedialog.askopenfilename(title="Choisir le chemin du fichier json d'entrainement (train)",
                                          filetypes=(("json files", ".json"),))

        root.destroy()
        self.train_file_name = os.path.basename(self.train_path)
        return self.train_path, self.train_file_name

    def val_path_picker(self):
        root = tk.Tk()
        root.withdraw()

        self.val_path = filedialog.askopenfilename(title="Choisir le chemin du fichier json de test (val)",
                                             filetypes=(("json files", ".json"),))
        self.val_file_name = os.path.basename(self.val_path)
        root.destroy()
        return self.val_path, self.val_file_name

    def optionmenu_callback(self, choice):
        print("optionmenu dropdown clicked:", choice)

    def train_button_callback(self):
        print("training has begun !")

    def get(self):
        path = []
        if self.import_train_button:
            path.append(self.import_train_button.cget("text"))

class ModelUserFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        # Variables globales
        self.img_path = ""
        self.img_file_name = os.path.basename(self.img_path)

        self.model_predictor = ModelPredictor()

        # Titre du frame
        self.title = ctk.CTkLabel(self, text="Image analyzer", fg_color="white", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=2)

        self.import_train_button = ctk.CTkButton(self, text="Import image", command=self.img_path_picker)
        self.import_train_button.grid(column=0, row=1, padx=10, pady=10, sticky="w", columnspan=1)

    def img_path_picker(self):
        root = tk.Tk()
        root.withdraw()

        self.img_path = filedialog.askopenfilename(title="choisir une image à analyser",
                                          filetypes=(("images files", ".jpg"),))

        root.destroy()
        self.img_file_name = os.path.basename(self.img_path)
        areas, ratio, image_name = self.model_predictor.analyze_image(self.img_path)

        # Afficher les résultats dans la console (ou dans l'interface si tu veux)
        print(f"Aires calculées (pixels): {areas}")
        print(f"Rapport broken/full : <{ratio:.2f}%> pour l'image : <{image_name}>.")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("my app")
        self.geometry("600x500")
        #self.grid_columnconfigure(0, weight=1)
        #self.grid_rowconfigure(0, weight=1)

        # Frame de l'interface pour l'initialisation
        self.model_trainer_frame = ModelTrainerFrame(self)
        self.model_trainer_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nw")

        # Frame de l'interface pour l'utilisation
        self.model_user_frame = ModelUserFrame(self)
        self.model_user_frame.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nw")

        #self.button = ctk.CTkButton(self, text="Train !", command=self.button_callback)
        #self.button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    def button_callback(self):
        print("train path selected:", self.model_trainer_frame.train_file_name)
        print("test path selected:", self.model_trainer_frame.val_file_name)

#app = App()
#app.mainloop()