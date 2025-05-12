import tkinter as tk
import random
from axis_aligned_rect import AdaboostRectangleLearner

# Canvas settings
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 600
POINT_RADIUS = 1.5

def to_canvas_coords(x, y):
    return x * CANVAS_WIDTH, CANVAS_HEIGHT - y * CANVAS_HEIGHT

class AdaBoostVisualizer:
    def __init__(self, master):
        self.master = master
        master.title("Axis-Aligned Rectangles | AdaBoost + Bagging")

        self.canvas = tk.Canvas(master, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
        self.canvas.pack()
        
        # --- Main horizontal frame ---
        control_frame = tk.Frame(master)
        control_frame.pack(fill='x', pady=5)

        # --- Create three side-by-side sub-frames ---
        inputs_frame = tk.Frame(control_frame)
        inputs_frame.pack(side='left', padx=20)

        buttons_frame = tk.Frame(control_frame)
        buttons_frame.pack(side='left', padx=20)

        outputs_frame = tk.Frame(control_frame)
        outputs_frame.pack(side='left', padx=20, anchor='n')

        # --- Inputs section ---
        self.dataset_noise_label = tk.Label(inputs_frame, text="Dataset noise:", anchor='e')
        self.dataset_noise_label.grid(row=0, column=0, sticky='e', pady=2)

        self.dataset_noise_entry = tk.Entry(inputs_frame, width=5)
        self.dataset_noise_entry.insert(tk.END, str(0))
        self.dataset_noise_entry.grid(row=0, column=1, sticky='w', pady=2)

        self.d_value_label = tk.Label(inputs_frame, text="d:", anchor='e')
        self.d_value_label.grid(row=1, column=0, sticky='e', pady=2)

        self.d_value_entry = tk.Entry(inputs_frame, width=5)
        self.d_value_entry.insert(tk.END, str(0.005))
        self.d_value_entry.grid(row=1, column=1, sticky='w', pady=2)

        self.boosting_rounds_label = tk.Label(inputs_frame, text="# boosting rounds:", anchor='e')
        self.boosting_rounds_label.grid(row=2, column=0, sticky='e', pady=2)

        self.boosting_rounds_entry = tk.Entry(inputs_frame, width=5)
        self.boosting_rounds_entry.insert(tk.END, str(1))
        self.boosting_rounds_entry.grid(row=2, column=1, sticky='w', pady=2)

        self.bagging_rounds_label = tk.Label(inputs_frame, text="# bagging rounds:", anchor='e')
        self.bagging_rounds_label.grid(row=3, column=0, sticky='e', pady=2)

        self.bagging_rounds_entry = tk.Entry(inputs_frame, width=5)
        self.bagging_rounds_entry.insert(tk.END, str(1))
        self.bagging_rounds_entry.grid(row=3, column=1, sticky='w', pady=2)

        # --- Buttons section ---
        self.generate_button = tk.Button(buttons_frame, width=14, text="Generate & Boost", command=self.generate)
        self.generate_button.pack(pady=2)

        self.hide_concept_button = tk.Button(buttons_frame, width=14, text="Hide Concept", command=self.toggle_concept_rectangles)
        self.hide_concept_button.pack(pady=2)

        self.hide_hypothesis_button = tk.Button(buttons_frame, width=14, text="Hide Hypothesis", command=self.toggle_hypothesis_rectangles)
        self.hide_hypothesis_button.pack(pady=2)

        self.hide_weak_button = tk.Button(buttons_frame, width=14, text="Hide Weak Learners", command=self.toggle_weak_rectangles)
        self.hide_weak_button.pack(pady=2)

        # --- Outputs section ---
        self.true_error = tk.Label(outputs_frame, text="True error: 0", anchor='w')
        self.true_error.pack(anchor='w', pady=2)

        self.test_error = tk.Label(outputs_frame, text="Test error: 0", anchor='w')
        self.test_error.pack(anchor='w', pady=2)

        self.t_positives = tk.Label(outputs_frame, text="True positives: 0", anchor='w')
        self.t_positives.pack(anchor='w', pady=2)

        self.f_positives = tk.Label(outputs_frame, text="False positives: 0", anchor='w')
        self.f_positives.pack(anchor='w', pady=2)

        self.concept_rectangle_visible = True  # Initially, rectangles are visible
        self.hypothesis_rectangle_visible = True  # Initially, rectangles are visible
        self.weak_rectangles_visible = False  # Hide weak learner until boosting is added

        self.data = []
        self.classifiers = []
        self.learner = AdaboostRectangleLearner()
        self.generate()

    def generate(self):
        self.generating = True
        self.update()
        self.generating = False
        
    def toggle_weak_rectangles(self):
        # Toggle the visibility of weak learner rectangles
        self.weak_rectangles_visible = not self.weak_rectangles_visible
        self.update()
        
    def toggle_concept_rectangles(self):
        # Toggle the visibility of rectangles
        self.concept_rectangle_visible = not self.concept_rectangle_visible
        self.update()

    def toggle_hypothesis_rectangles(self):
        # Toggle the visibility of rectangles
        self.hypothesis_rectangle_visible = not self.hypothesis_rectangle_visible
        self.update()

    def update(self):
        noise_rate = float(self.dataset_noise_entry.get())
        d_value = float(self.d_value_entry.get())
        bagging_rounds = int(self.bagging_rounds_entry.get())
        boosting_rounds = int(self.boosting_rounds_entry.get())

        self.canvas.delete("all")
        
        if self.generating:
            # Generate new dataset
            self.data, self.concept_rect = self.learner.generate_data(noise_rate=noise_rate)  # Get the new dataset
            self.bagged_classifiers = self.learner.adaboost_with_bagging(self.data, num_bags=bagging_rounds, num_rounds=boosting_rounds, d=d_value)  # Run AdaBoost on the data
        
            self.hypo_rectangle = self.learner.get_consensus_rectangle(self.bagged_classifiers)
            self.true_error.config(text=f"True error: {self.learner.true_error(self.concept_rect, self.hypo_rectangle)[0]:.3f}")
            self.test_error.config(text=f"Test error: {self.learner.test_error(self.data, self.hypo_rectangle):.3f}")
            
            positives = [(x, y) for (x, y, l) in self.data if l == 1]
            self.t_positives.config(text=f"True positives: {len([(x, y) for (x, y) in positives if self.learner.classify(self.concept_rect, x, y) == 1])}")
            self.f_positives.config(text=f"False positives: {len([(x, y) for (x, y) in positives if self.learner.classify(self.concept_rect, x, y) == 0])}")
        
        self.draw()

    def draw(self):
        # Draw data points
        samples = random.sample(self.data, k=10000)
        for x, y, label in samples:
            cx, cy = to_canvas_coords(x, y)
            color = "green" if label == 1 else "red"
            self.canvas.create_oval(cx - POINT_RADIUS, cy - POINT_RADIUS,
                                    cx + POINT_RADIUS, cy + POINT_RADIUS,
                                    fill=color, outline=color)
        
        # Draw learned rectangles
        if self.weak_rectangles_visible:
            for classifiers in self.bagged_classifiers:
                for i, (rect, pol, alpha) in enumerate(classifiers):
                    if not rect:
                        continue
                    xmin, xmax, ymin, ymax = rect
                    x1, y1 = to_canvas_coords(xmin, ymax)
                    x2, y2 = to_canvas_coords(xmax, ymin)

                    thickness = max(1, int(alpha * 5))
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=thickness)

        if self.generating:
            self.hypo_rectangle = self.learner.get_consensus_rectangle(self.bagged_classifiers)

        if self.concept_rectangle_visible:
            # Draw the true concept (gray dashed) centered at (shift_x, shift_y)
            xmin, xmax, ymin, ymax = self.concept_rect
            x1, y1 = to_canvas_coords(xmin, ymax)
            x2, y2 = to_canvas_coords(xmax, ymin)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray", dash=(4, 2), width=2)
        
        if self.hypo_rectangle and self.hypothesis_rectangle_visible:
            xmin, xmax, ymin, ymax = self.hypo_rectangle
            # Draw the hypothesis rectangle
            x1, y1 = to_canvas_coords(xmin, ymax)
            x2, y2 = to_canvas_coords(xmax, ymin)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="purple", width=3, dash=(5, 3))


# Run app
root = tk.Tk()
app = AdaBoostVisualizer(root)
root.mainloop()
