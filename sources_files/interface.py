import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from machine_learning_model.useModel import ModelPredictor
from machine_learning_model.trainModel import ModelTrainer


class ModelTrainerFrame(ctk.CTkFrame):
    """
    Frame pour l'entraînement du modèle
    TODO améliorer le code en rendant le placement des boutons des params dynamique (au lieu de les 'hard codés')
    """
    def __init__(self, master):
        super().__init__(master)

        # Variables globales
        self.coco_path = ""
        self.coco_file_name = os.path.basename(self.coco_path)

        self.dataset_path = ""
        self.dataset_file_name = os.path.basename(self.dataset_path)

        self.model_trainer = ModelTrainer() # Instance de ModelTrainer

        # Définir les paramètres par défaut de l'interface
        self.params = {
            "num_iterations": 300,
            "num_classes": 1,
            "roihead_batch_size": 512,
            "real_batch_size": 2,
            "learning_rate": 0.00025,
            "model_device": "cpu"
        }

        self.grid_columnconfigure(0, weight=1)

        # Titre du frame
        self.title = ctk.CTkLabel(self, text="Model Trainer", fg_color="white", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=2)

        # Boutons d'import/export des dataset
        # Importer un fichier json avec les annotations pour entraînement
        self.import_train_button = ctk.CTkButton(self, text="Import COCO annotations", command=self.coco_path_picker)
        self.import_train_button.grid(column=0, row=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Importer un fichier json avec les annotations pour test
        self.import_test_button = ctk.CTkButton(self, text="Import training dataset", command=self.dataset_path_picker)
        self.import_test_button.grid(column=0, row=2, padx=10, pady=10, sticky="ew", columnspan=1)

        # Afficheur de nom des fichiers sélectionnés (pour train et test)
        self.train_path_label = ctk.CTkLabel(self, text=self.coco_file_name, fg_color="white", corner_radius=6)
        self.train_path_label.grid(column=1, row=1, padx=10, pady=10, sticky="ew", columnspan=1)

        self.val_path_label = ctk.CTkLabel(self, text=self.dataset_file_name, fg_color="white", corner_radius=6)
        self.val_path_label.grid(column=1, row=2, padx=10, pady=10, sticky="ew", columnspan=1)

        # Gestion des paramètres :
        # param1:Nombre d'itérations
        self.param1 = ctk.CTkLabel(self, text="Nombre d'itération", fg_color="white", corner_radius=6)
        self.param1.grid(row=3, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param1_option = ctk.CTkOptionMenu(
            self,
            values=["300", "500", "800", "1000"],
            command=lambda choice: self.optionmenu_callback("num_iterations", choice)
        )
        self.param1_option.grid(row=3, column=1, padx=10, pady=10, sticky="w", columnspan=1)

        # param2:Nombre de classe
        self.param2 = ctk.CTkLabel(self, text="Nombre de classe", fg_color="white", corner_radius=6)
        self.param2.grid(row=4, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param2_option = ctk.CTkOptionMenu(
            self,
            values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            command=lambda choice: self.optionmenu_callback("num_classes", choice)
        )
        self.param2_option.grid(row=4, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # param3:Batch size par image "RoIHead batch size"
        self.param3 = ctk.CTkLabel(self, text="RoIHead batch size", fg_color="white", corner_radius=6)
        self.param3.grid(row=5, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param3_option = ctk.CTkOptionMenu(
            self,
            values=["128", "256", "384", "512", "640", "768", "896"],
            command=lambda choice: self.optionmenu_callback("real_batch_size", choice)
        )
        self.param3_option.set("512")
        self.param3_option.grid(row=5, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # param4:real batch size
        self.param4 = ctk.CTkLabel(self, text="Real batch size", fg_color="white", corner_radius=6)
        self.param4.grid(row=6, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param4_option = ctk.CTkOptionMenu(
            self,
            values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            command=lambda choice: self.optionmenu_callback("real_batch_size", choice)
        )
        self.param4_option.set("2")
        self.param4_option.grid(row=6, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # param5:learning rate
        self.param5 = ctk.CTkLabel(self, text="Learning rate LR", fg_color="white", corner_radius=6)
        self.param5.grid(row=7, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param5_option = ctk.CTkOptionMenu(
            self,
            values=["0.00020", "0.00025", "0.00030", "0.00035", "0.00040", "0.00045", "0.00050", "0.00055", "0.00060", "0.00065"],
            command=lambda choice: self.optionmenu_callback("learning_rate", choice)
        )
        self.param5_option.set("0.00025")
        self.param5_option.grid(row=7, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # param6:model device
        self.param6 = ctk.CTkLabel(self, text="Model device", fg_color="white", corner_radius=6)
        self.param6.grid(row=8, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param6_option = ctk.CTkOptionMenu(
            self,
            values=["cpu", "cuda"],
            command=lambda choice: self.optionmenu_callback("model_device", choice)
        )
        self.param6_option.grid(row=8, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Begin training button
        self.train_button = ctk.CTkButton(self, text="Train !", corner_radius=6, command=self.train_button_callback)
        self.train_button.grid(row=9, column=0, padx=10, pady=10, sticky="ew", columnspan=2)


    def coco_path_picker(self):
        root = tk.Tk()
        root.withdraw()

        self.coco_path = filedialog.askopenfilename(title="Choisir le chemin du fichier json d'entrainement (train)",
                                                    filetypes=(("json files", ".json"),)
                                                    )
        root.destroy()

        self.coco_file_name = os.path.basename(self.coco_path) # MAJ du nom du fichier
        self.train_path_label.configure(text=self.coco_file_name) # MAJ du plugin label pour qu'il affiche le nom du fichier

        return self.coco_path, self.coco_file_name

    def dataset_path_picker(self):
        root = tk.Tk()
        root.withdraw()

        self.dataset_path = filedialog.askdirectory(title="Choisir le chemin vers le training dataset")

        root.destroy()

        self.dataset_file_name = os.path.basename(self.dataset_path)
        self.val_path_label.configure(text=self.dataset_file_name)

        return self.dataset_path, self.dataset_file_name

    def optionmenu_callback(self, param_name, choice):
        """
        Callback générique pour tous les paramètres\n
        :param param_name: Nom du paramètre à maj
        :param choice: Valeur sélectionnée dans le menu
        """
        #print("optionmenu dropdown clicked:", choice)
        print(f"Le paramètre '{param_name}' a été mis à jour avec la valeur : {choice}")

        # TODO mettre dynamique ... éventuellement
        if param_name in ["num_iterations", "num_classes", "roihead_batch_size", "real_batch_size"]:
            self.params[param_name] = int(choice)
        elif param_name == "learning_rate":
            self.params[param_name] = float(choice) # LR est un float, on pourrait conserver des int en multipliant, mais ce n'est pas tant nécessaire ici
        elif param_name == "model_device":
            self.params[param_name] = choice # model device est un string ('cpu' ou 'cuda')

        #print("Params actuels :", self.params)

    def train_button_callback(self):
        print("Début de l'entraînement selon les paramètres suivant:", self.params)
        self.model_trainer = ModelTrainer(**self.params)
        self.model_trainer.begin_training(self.coco_path, self.dataset_path)

    def get(self):
        return None
        #path = []
        #if self.import_train_button:
        #    path.append(self.import_train_button.cget("text"))

class ModelUserFrame(ctk.CTkFrame):
    """
    Frame pour l'utilisation du modèle
    """
    def __init__(self, master):
        super().__init__(master)

        # Variables globales
        self.img_path = ""
        self.img_file_name = os.path.basename(self.img_path)

        self.result_text = ""

        self.grid_columnconfigure(1, weight=1)

        self.model_predictor = ModelPredictor() # Instance de ModelPredictor

        # Titre du frame
        self.title = ctk.CTkLabel(self, text="Image analyzer", fg_color="white", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=2)

        # Import du fichier image
        self.import_train_button = ctk.CTkButton(self, text="Import image", command=self.img_path_picker)
        self.import_train_button.grid(column=0, row=1, padx=10, pady=10, sticky="w", columnspan=1)

        # Affichage du nom du fichier image
        self.img_path_label = ctk.CTkLabel(self, text=self.img_file_name, fg_color="white", corner_radius=6)
        self.img_path_label.grid(column=1, row=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Entamer la prédiction avec le modèle
        self.begin_pred_button = ctk.CTkButton(self, text="Go !", command=self.begin_pred)
        self.begin_pred_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

        """
        # Afficher l'image avec les prédictions, stand-by
        self.pred_image_canvas = ctk.CTkCanvas(self, width=300, height=300, corner_radius=6)
        self.pred_image_canvas.grid(column=0, row=3, padx=10, pady=10, sticky="ew", columnspan=2)

        self.img = Image.open("sources_files/assets/images/Alius_FR.png")
        self.resized_image = self.img.resize((300, 300))
        self.image = ImageTk.PhotoImage(self.resized_image)
        self.my_image = self.pred_image_canvas.create_image(300, 300, image=self.image, anchor="nw")
        """
        # Afficher les résultats souhaités
        self.result_textbox = ctk.CTkTextbox(master=self, width=100, corner_radius=6)
        self.result_textbox.grid(row=4, column=0, sticky="ew", columnspan=2)

    def img_path_picker(self):
        """
        Ouvre une boîte de dialogue vers le chemin de l'image à passer dans le modèle. \n
        Prend le nom du fichier et l'applique au label 'img_file_name' dans l'interface.
        :return: None
        """
        root = tk.Tk()
        root.withdraw()

        self.img_path = filedialog.askopenfilename(title="choisir une image à analyser",
                                          filetypes=(("images files", ".jpg"),))

        root.destroy()
        self.img_file_name = os.path.basename(self.img_path)
        self.img_path_label.configure(text=self.img_file_name)

    def begin_pred(self):
        """
        Appellée à l'aide de 'begin_pred_button' et lance la prédiction sur une image avec le modèle entraîné.
        :return: String des données souhaitées.
        """
        class_names, areas, ratio, image_name = self.model_predictor.analyze_image(self.img_path)
        # Afficher les résultats, pourrait être changé pour choisir dynamiquement ce que l'on veut extraire de l'image.
        # Par exemple, des "radios buttons" pour savoir le RQD, composition de l'échantillon, nbr de faille, etc.
        self.result_text = (f"Résultats pour l'image : <{image_name}>\n"
                                f"=======================================\n"
                                f"Rapport broken/full    : <{ratio:.2f}%>\n"
                                f"Aire de {class_names[0]}: {areas['cz']} px\n"
                                f"Aire de {class_names[1]}: {areas['core_empty_or_broken']} px\n"
                                f"Aire de {class_names[2]}: {areas['core_full']} px\n"
                            )
        self.result_textbox.insert("0.0", self.result_text)
        return self.result_text
        #print(f"Aires calculées (pixels): {areas}")
        #print(f"Rapport broken/full : <{ratio:.2f}%> pour l'image : <{image_name}>.")
        # TODO 2 : exporter résultats vers excel éventuellement

class App(ctk.CTk):
    """
    Gestionnaire de l'application
    """
    def __init__(self):
        super().__init__()

        self.title("my app")
        self.geometry("700x500")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame de l'interface pour l'initialisation
        self.model_trainer_frame = ModelTrainerFrame(self)
        self.model_trainer_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nw")

        # Frame de l'interface pour l'utilisation
        self.model_user_frame = ModelUserFrame(self)
        self.model_user_frame.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nwe")


    def button_callback(self): # Pour tests seulement
        print("train path selected:", self.model_trainer_frame.coco_file_name)
        print("test path selected:", self.model_trainer_frame.dataset_file_name)


#app = App() # Appelé dans main.py
#app.mainloop()