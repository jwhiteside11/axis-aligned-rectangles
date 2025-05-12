import tkinter as tk
from axis_aligned_rect import RectangleLearner

# Parameters
RECTANGLE_SIZE = 400
POINT_RADIUS = 1

# Convert from normalized (0-1) to canvas coords
def to_canvas_coords(x, y):
  return (x * RECTANGLE_SIZE + RECTANGLE_SIZE // 2, RECTANGLE_SIZE - y * RECTANGLE_SIZE + RECTANGLE_SIZE // 2)

# GUI app
class RectangleVisualizer(tk.Tk):
  def __init__(self):
    super().__init__()
    self.title("Axis-Aligned Rectangle Learner")
    
    self.m = 1 # number of positive learning examples
    self.n = 2000 # number of test points

    # Canvas
    self.canvas = tk.Canvas(self, width=RECTANGLE_SIZE*2, height=RECTANGLE_SIZE*2, bg="white")
    self.canvas.pack()

    # Bottom controls
    self.control_frame = tk.Frame(self)
    self.control_frame.pack(pady=10)

    # Input
    tk.Label(self.control_frame, text="m =").grid(row=0, column=1)
    self.input_m = tk.Entry(self.control_frame, width=6)
    self.input_m.grid(row=0, column=2)
    self.input_m.insert(0, str(self.m)) # Set initial value in UI

    # Button to trigger update
    self.update_button = tk.Button(self.control_frame, text="Generate & Learn", command=self.update)
    self.update_button.grid(row=0, column=3, padx=10)

    # Output text label (column 7+ to be to the right of buttons)
    self.output_label = tk.Label(self.control_frame, text="", fg="blue", anchor="w", width=25)
    self.output_label.grid(row=0, column=4, padx=10)

    # Button to trigger update
    self.add_point_button = tk.Button(self.control_frame, text="Add point", command=self.add_point)
    self.add_point_button.grid(row=0, column=5, padx=10)

    self.learner = RectangleLearner(self.n, 1)
    self.update()

  def update(self):
    try:
      self.m = max(int(self.input_m.get()), 1)
    except:
      self.m = 1
      
    self.learner.generate(self.n, self.m)
    self.draw()

  def add_point(self):
    self.learner.add_point()
    self.m = len(self.learner.training_data)
    self.input_m.delete(0, tk.END) # Set value in UI
    self.input_m.insert(0, str(self.m)) # Set value in UI
    self.draw()

  def draw(self):
    self.canvas.delete("all")
    
    # Draw accuracy score
    self.output_label.config(text=f"Accuracy: {self.learner.accuracy:.6f}")

    # Draw test points
    for x, y in self.learner.testing_data:
      if self.learner.classify((x, y), self.learner.rectangle) == 0:
        continue
      cx, cy = to_canvas_coords(x, y)
      color = "green" if self.learner.classify((x, y), self.learner.learned_rectangle) == 1 else "red"
      self.canvas.create_oval(cx - POINT_RADIUS, cy - POINT_RADIUS, cx + POINT_RADIUS, cy + POINT_RADIUS, fill=color, outline=color)
    
    # Draw training points
    for x, y in self.learner.training_data:
      cx, cy = to_canvas_coords(x, y)
      color = "black"
      self.canvas.create_oval(cx - POINT_RADIUS*3, cy - POINT_RADIUS*3, cx + POINT_RADIUS*3, cy + POINT_RADIUS*3, fill=color, outline=color)

    # Draw learned rectangle
    xmin, xmax, ymin, ymax = self.learner.learned_rectangle
    x1, y1 = to_canvas_coords(xmin, ymax)
    x2, y2 = to_canvas_coords(xmax, ymin)
    self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=4)

    # Draw known rectangle
    xmin, xmax, ymin, ymax = self.learner.rectangle
    x1, y1 = to_canvas_coords(xmin, ymax)
    x2, y2 = to_canvas_coords(xmax, ymin)
    self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2)

# Run the app
app = RectangleVisualizer()
app.mainloop()
