import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image

from sources_files.useModel import ModelPredictor
from sources_files.trainModel import ModelTrainer


class ModelTrainerFrame(ctk.CTkFrame):
    """
    Frame pour l'entraînement du modèle
    TODO améliorer le code en rendant le placement des boutons des params dynamique (au lieu de les 'hard codés')
    """
    def __init__(self, master):
        super().__init__(master)

        # Variables globales
        # Chemin vers les annotations json (format COCO seulement)
        self.coco_path = ""
        self.coco_file_name = os.path.basename(self.coco_path)

        # Chemin vers le dossier contenant le dataset (images seulement)
        self.dataset_path = ""
        self.dataset_file_name = os.path.basename(self.dataset_path)

        # Nom du modèle entraîné. Input par l'utilisateur, utile pour entraîner plusieurs modèles différents
        self.model_name_default = "modèleNom_model"
        self.model_name = ""

        #self.num_iterations = "0"

        self.model_trainer = ModelTrainer # Instance de ModelTrainer

        # Définir les paramètres par défaut de l'interface
        self.params = {
            "num_iterations": 500,
            "num_classes": 1,
            "roihead_batch_size": 512,
            "real_batch_size": 2,
            "learning_rate": 0.00025,
            "model_device": "cpu"
        }

        self.grid_columnconfigure(0, weight=1)

        # Titre du frame
        self.title = ctk.CTkLabel(self, text="Entraîneur de modèle", fg_color="white", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=2)

        # Boutons d'import/export des dataset
        # Importer un fichier json avec les annotations pour entraînement
        self.import_train_button = ctk.CTkButton(self, text="Importer annotations COCO", command=self.coco_path_picker)
        self.import_train_button.grid(column=0, row=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Importer un fichier json avec les annotations pour test
        self.import_test_button = ctk.CTkButton(self, text="Importer données d'entraînement", command=self.dataset_path_picker)
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

        #self.param1_option = ctk.CTkEntry(self, placeholder_text="ex: 2000")
        self.param1_option = ctk.CTkOptionMenu(
            self,
            values=["500", "1000", "2000", "5000", "10000", "20000", "40000"],
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
            command=lambda choice: self.optionmenu_callback("roihead_batch_size", choice)
        )
        self.param3_option.set("512")
        self.param3_option.grid(row=5, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # param4:real batch size
        self.param4 = ctk.CTkLabel(self, text="Real batch size", fg_color="white", corner_radius=6)
        self.param4.grid(row=6, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param4_option = ctk.CTkOptionMenu(
            self,
            values=["1", "2", "3", "4", "8", "16"],
            command=lambda choice: self.optionmenu_callback("real_batch_size", choice)
        )
        self.param4_option.set("2")
        self.param4_option.grid(row=6, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # param5:learning rate
        self.param5 = ctk.CTkLabel(self, text="Learning rate LR", fg_color="white", corner_radius=6)
        self.param5.grid(row=7, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        self.param5_option = ctk.CTkOptionMenu(
            self,
            values=["0.001", "0.00020", "0.00025", "0.00030", "0.00035", "0.00040", "0.00045", "0.00050", "0.00055", "0.00060", "0.00065"],
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

        # Entrée pour l'utilisateur lui permettant de setter le nom d'un modèle
        self.model_name_entry = ctk.CTkEntry(self, placeholder_text=self.model_name_default)
        self.model_name_entry.grid(row=9, column=0, padx=10, pady=10, sticky="ew", columnspan=1)

        # Bouton de sauvegarde du nom du modèle
        self.model_name_entry_btn = ctk.CTkButton(self, text="Sauvegarder le nom", corner_radius=6, command=self.save_model_name)
        self.model_name_entry_btn.grid(row=9, column=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Begin training button
        self.train_button = ctk.CTkButton(self, text="Entraîner un modèle avec ces options", corner_radius=6, command=self.train_button_callback)
        self.train_button.grid(row=10, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

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
            self.params[param_name] = float(choice) # LR est un float
        elif param_name == "model_device":
            self.params[param_name] = choice # model device est un string ('cpu' ou 'cuda')

        #print("Params actuels :", self.params)

    def train_button_callback(self):
        print("Début de l'entraînement selon les paramètres suivant:", self.params)
        self.model_trainer = ModelTrainer(**self.params)
        self.model_trainer.begin_training(self.coco_path, self.dataset_path, self.model_name)

    def save_model_name(self):
        self.model_name = self.model_name_entry.get()
        print("Nom du modèle:", self.model_name)

    def dummy_callback(self, thing):
        """
        Dummy callback pour tests
        :param thing:
        """
        print(f"Something is happening ! {thing}")

class ModelUserFrame(ctk.CTkFrame):
    """
    Frame pour l'utilisation du modèle
    """
    def __init__(self, master):
        super().__init__(master)
        # TODO standardiser le nom des variables, c'est un shitshow >:(
        # Variables globales
        self.img_path = "" # Chemin vers l'image à analyser
        self.img_file_name = os.path.basename(self.img_path) # Nom seul de l'image à analyser

        self.folder_path = ""
        self.folder_file_name = os.path.basename(self.folder_path)

        self.modele_dir = "" # Chemin vers le directory du modèle de prédiction
        self.model_selected = False
        self.model_dir_name = os.path.basename(self.modele_dir) # Nom seul du modèle de prédiction

        self.num_class = "" # Nombre de classe (d'annotation) à passer au constructeur de 'useModel'

        self.result_text = "" # Sortie dans la boîte de texte de l'analyse

        self.grid_columnconfigure(1, weight=1) # Pour la mise en page

        # Empty fields en attendant l'appel de model_dir_path_picker
        self.model_predictor = None
        self.methods_dict = {}
        self.methods_dict_keys = [] # =list(self.methods_dict.keys())
        self.selected_method = self.methods_dict_keys[:-1]

        # Titre du frame
        self.title = ctk.CTkLabel(self, text="Prédicteur", fg_color="white", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=2)

        # Import de l'image
        self.import_train_button = ctk.CTkButton(self, text="Importer une image", command=self.img_path_picker)
        self.import_train_button.grid(column=0, row=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Affichage du nom de l'image
        self.img_path_label = ctk.CTkLabel(self, text=self.img_file_name, fg_color="white", corner_radius=6)
        self.img_path_label.grid(column=1, row=1, padx=10, pady=10, sticky="ew", columnspan=1)

        # Import du fichier d'images
        self.import_img_folder_button = ctk.CTkButton(self, text="Importer un fichier d'images", command=self.folder_path_picker)
        self.import_img_folder_button.grid(column=0, row=2, padx=10, pady=10, sticky="ew", columnspan=1)

        # Affichage du fichier d'images
        self.img_folder_path_label = ctk.CTkLabel(self, text=self.folder_file_name, fg_color="white", corner_radius=6)
        self.img_folder_path_label.grid(column=1, row=2, padx=10, pady=10, sticky="ew", columnspan=1)

        # Label pour le nombre de classe à passer au constructeur
        self.choose_number_of_class_label = ctk.CTkLabel(self, text="Nombre de classe", fg_color="white", corner_radius=6)
        self.choose_number_of_class_label.grid(column=0, row=3, padx=10, pady=10, sticky="ew", columnspan=1)

        # Input pour le nombre de classe à passer au constructeur
        self.choose_number_of_class_entry = ctk.CTkEntry(self, placeholder_text="1, 2, 3, ...", fg_color="white", corner_radius=6)
        self.choose_number_of_class_entry.grid(column=1, row=3, padx=10, pady=10, sticky="ew", columnspan=1)

        # Import du directory du modèle
        self.import_model_dir_button = ctk.CTkButton(self, text="Importer un modèle", command=self.model_dir_path_picker)
        self.import_model_dir_button.grid(column=0, row=4, padx=10, pady=10, sticky="ew", columnspan=1)

        # Affichage du nom du model directory
        self.model_dir_label = ctk.CTkLabel(self, text=self.model_dir_name, fg_color="white", corner_radius=6)
        self.model_dir_label.grid(column=1, row=4, padx=10, pady=10, sticky="ew", columnspan=1)

        # Choix de la méthode d'analyse
        self.choose_method_button = ctk.CTkLabel(self, text="Méthode d'analyse :", fg_color="white", corner_radius=6)
        self.choose_method_button.grid(column=0, row=5, padx=10, pady=10, sticky="ew", columnspan=1)

        # Options des méthodes d'analyse
        self.choose_methode_optionmenu = ctk.CTkOptionMenu(self,
            values=self.methods_dict_keys,
            command=self.method_selected
           )
        self.choose_methode_optionmenu.grid(column=1, row=5, padx=10, pady=10, sticky="ew", columnspan=1)

        # TODO afficher un texte qui montre ce que la méthode d'analyse fait

        # Entamer la prédiction avec le modèle
        self.begin_pred_button = ctk.CTkButton(self, text="Go !", command=self.run_selected_method)
        self.begin_pred_button.grid(row=6, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

        # Afficher les résultats souhaités
        #self.result_textbox = ctk.CTkTextbox(self, width=100, corner_radius=6)
        #self.result_textbox.grid(row=7, column=0, sticky="ew", columnspan=2)

    def img_path_picker(self): # TODO Serait cool de pouvoir passer plusieurs images de suite
        """
        Ouvre une boîte d'explorateur de fichier vers le chemin de l'image à passer dans le modèle. \n
        Prend le nom du fichier et l'applique au label 'img_file_name' dans l'interface.
        """
        root = tk.Tk()
        root.withdraw()

        self.img_path = filedialog.askopenfilename(title="choisir une image à analyser",
                                                   filetypes=(("images files", ".jpg"),),
                                                   initialdir="machine_learning_model/datasets"
                                                   )
        root.destroy()
        self.img_file_name = os.path.basename(self.img_path)
        self.img_path_label.configure(text=self.img_file_name)

    def folder_path_picker(self):
        """

        :return:
        """
        root = tk.Tk()
        root.withdraw()

        self.folder_path = filedialog.askdirectory(
            mustexist=True,
            initialdir="machine_learning_model/datasets",
            title="choisir un fichier d'images à analyser"
        )
        root.destroy()
        self.folder_file_name = os.path.basename(self.folder_path)
        self.img_folder_path_label.configure(text=self.folder_file_name) #maj du texte label

    def model_dir_path_picker(self):
        """
        Ouvre une boîte d'explorateur de fichier vers le chemin du directory du modèle à utilisé \n
        Prend le nom du fichier et l'applique au label 'img_file_name' dans l'interface.
        """
        root = tk.Tk()
        root.withdraw()

        self.modele_dir = filedialog.askdirectory(
            mustexist=True,
            title="Choisir un modèle d'analyse",
            initialdir="machine_learning_model/trained_models"
        )
        self.model_selected = True if self.modele_dir else False

        if self.model_selected:
            self.num_class = int(self.choose_number_of_class_entry.get())
            self.model_predictor = ModelPredictor(model_dir=self.modele_dir, num_classes=self.num_class)  # Instance de ModelPredictor lorsqu'un model_dir existe

            self.methods_dict = self.model_predictor.get_methods() # Initialisation du dictionnaire qui contient les méthodes d'analyses

            self.methods_dict_keys = list(self.methods_dict.keys()) # Initialisation de la liste des clés
            self.choose_methode_optionmenu.configure(values=self.methods_dict_keys)

            self.selected_method = list(self.methods_dict.keys())[0] # On choisi la première méthode d'analyse par défaut
            print("Un modèle a été sélectionné :", self.modele_dir)

        root.destroy()
        self.model_dir_name = os.path.basename(self.modele_dir)
        self.model_dir_label.configure(text=self.model_dir_name)

    def method_selected(self, choice):
        """
        Associe la méthode choisi via le menu à la variable globale 'selected_method'.
        :param choice: Choix de la méthode d'analyse, tiré des méthodes dans 'useModel.py'
        :return:
        """
        self.selected_method = choice
        print(f"Méthode sélectionnée : {self.selected_method}")

    def run_selected_method(self):
        if self.img_path: # Si une image est selectionnée
            analysis_function = self.methods_dict[self.selected_method] # on trouve notre méthode dans le dict avec la clé 'self.selected_method'
            #output = analysis_function(self.img_path)
            analysis_function(self.img_path)

        elif self.folder_path:
            analysis_function = self.methods_dict[self.selected_method]
            #output = analysis_function(self.folder_path)
            analysis_function(self.folder_path)

            # ecrire code pour choix dynamique des fonctions, voir le stack overflow

            #print(f"Résultat de l'analyse : {result}") # results
        else:
            print("Aucune image sélectionnée")

    def begin_pred(self):
        """
        Ancienne méthode, gardée seulement pour les notes
        :param chosen_method: Méthode d'analyse utilisée dans la prédiction
        :return: String des données souhaitées.
        """

        #self.model_predictor = ModelPredictor(model_dir=self.modele_dir)  # Instance de ModelPredictor
        #self.method_dict = self.model_predictor.get_methods()

        class_names, class_area,  img_name = self.model_predictor.analyze_image(self.img_path)
        # Afficher les résultats, pourrait être changé pour choisir dynamiquement ce que l'on veut extraire de l'image.
        # Par exemple, des "radios buttons" pour savoir le RQD, composition de l'échantillon, nbr de faille, etc.
        self.result_text = (f"Résultats pour l'image : <{img_name}>\n"
                                f"=======================================\n"
                                f"Aire de {class_names[0]}: {class_area['cz']} px\n"
                            )
        #self.result_textbox.insert("0.0", self.result_text)
        return self.result_text
        #print(f"Aires calculées (pixels): {areas}")
        #print(f"Rapport broken/full : <{ratio:.2f}%> pour l'image : <{image_name}>.")

class App(ctk.CTk):
    """
    Gestionnaire de l'application
    """
    width = 800
    height = 550
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Titre et logo
        self.title("Analyseur de Carotte Alius")
        self.iconbitmap("sources_files/assets/images/Alius_logo_3.ico")

        # Géométrie de la fenêtre
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Image d'arrière plan
        self.bg_image = ctk.CTkImage(Image.open("sources_files/assets/images/Alius_logo_1.png"), size=(200, 200))
        self.bg_image_label = ctk.CTkLabel(self, image=self.bg_image, text="")
        self.bg_image_label.grid(row=0, column=1, columnspan=1, sticky="sew")

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