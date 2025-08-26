from sources_files import interface

def main():
    interface.ctk.set_appearance_mode("light")
    interface.ctk.set_default_color_theme("dark-blue")

    app = interface.App()
    app.mainloop()

if __name__ == "__main__":
    main()