import customtkinter as ctk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
from pathlib import Path

# CONFIGURATION
OUTPUT_DIR = Path("D:/aiml/assignment2/v3/data/clean/task2_outputs/")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ------------------------------
# APP WINDOW
app = ctk.CTk()
app.title("⚽ FIFA World Cup Prediction Dashboard")
app.geometry("1150x750")

# HEADER
header = ctk.CTkLabel(
    app,
    text="🏆 FIFA World Cup 2026 Prediction Dashboard",
    font=ctk.CTkFont(size=28, weight="bold"),
    text_color="#00BFFF"
)
header.pack(pady=20)

# ------------------------------
tabview = ctk.CTkTabview(app, width=1050, height=600, corner_radius=12)
tabview.pack(padx=20, pady=10, expand=True, fill="both")

tab_data = tabview.add("📊 Data Tables")
tab_plots = tabview.add("📈 Plots & Visuals")

# ------------------------------
status_label = ctk.CTkLabel(
    app,
    text="Select a prediction file or view the plots below.",
    font=ctk.CTkFont(size=16)
)
status_label.pack(pady=10)

# ------------------------------
table_label = ctk.CTkLabel(
    tab_data,
    text="Prediction Data",
    font=ctk.CTkFont(size=20, weight="bold"),
    text_color="#FFD700"
)
table_label.pack(pady=10)

# Dropdown selector for CSVs
file_selector = ctk.CTkOptionMenu(
    tab_data,
    values=[
        "predicted_finalists_2026.csv",
        "knockout_predictions.csv",
        "model_metrics.csv",
        "feature_importances.csv",
        "predictions.csv"
    ],
    width=280,
    height=40
)
file_selector.pack(pady=10)

# Treeview (table)
tree = ttk.Treeview(tab_data, show="headings")
tree.pack(fill="both", expand=True, padx=20, pady=10)

# Treeview style
style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview",
                background="#1e1e1e",
                fieldbackground="#1e1e1e",
                foreground="white",
                rowheight=28,
                font=("Segoe UI", 12))
style.configure("Treeview.Heading",
                font=("Segoe UI", 13, "bold"),
                foreground="#00BFFF",
                background="#202020")

# ------------------------------
# FUNCTIONS
def load_csv_file(*args):
    """Automatically load the selected CSV file."""
    filename = file_selector.get()
    filepath = OUTPUT_DIR / filename
    tree.delete(*tree.get_children())

    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            tree["columns"] = list(df.columns)
            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=180)
            for _, row in df.iterrows():
                tree.insert("", "end", values=list(row))
            status_label.configure(text=f"✅ Loaded: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
    else:
        status_label.configure(text=f"⚠️ File not found: {filename}")

# Bind dropdown change to auto-load
file_selector.configure(command=load_csv_file)

# ----------------------------
plot_label = ctk.CTkLabel(
    tab_plots,
    text="Model Evaluation Plots",
    font=ctk.CTkFont(size=20, weight="bold"),
    text_color="#FFD700"
)
plot_label.pack(pady=10)

canvas = ctk.CTkCanvas(tab_plots, bg="#1e1e1e", highlightthickness=0)
scroll_y = ctk.CTkScrollbar(tab_plots, orientation="vertical", command=canvas.yview)
frame_images = ctk.CTkFrame(canvas, fg_color="#1e1e1e")

frame_images.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=frame_images, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)
canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

def load_plot_images():
    """Load and display all .png images from OUTPUT_DIR."""
    for widget in frame_images.winfo_children():
        widget.destroy()

    image_files = list(OUTPUT_DIR.glob("*.png"))
    if not image_files:
        status_label.configure(text="⚠️ No .png plots found in output directory.")
        return

    for img_file in image_files:
        try:
            img = Image.open(img_file)
            img.thumbnail((900, 600))
            img_tk = ImageTk.PhotoImage(img)
            lbl = ctk.CTkLabel(frame_images, image=img_tk, text=img_file.name)
            lbl.image = img_tk  # prevent garbage collection
            lbl.pack(pady=20)
        except Exception as e:
            ctk.CTkLabel(frame_images, text=f"⚠️ Error loading {img_file.name}: {e}").pack()

    status_label.configure(text=f"✅ Loaded {len(image_files)} plot(s).")

# Auto-load plots on startup
load_plot_images()

# ------------------------------
exit_button = ctk.CTkButton(
    app,
    text="❌ Exit",
    command=app.destroy,
    fg_color="#B22222",
    hover_color="#FF4444",
    width=200,
    height=45,
    font=ctk.CTkFont(size=16, weight="bold")
)
exit_button.pack(pady=10)

# ------------------------------
if (OUTPUT_DIR / "predicted_finalists_2026.csv").exists():
    file_selector.set("predicted_finalists_2026.csv")
    load_csv_file()

# ------------------------------
app.mainloop()
