from manim import *
from scipy.integrate import solve_ivp
import random
import numpy as np
from scipy.integrate import odeint

class LorenzAttractor(ThreeDScene):
    """
    Animate the Butterfly Effect using the Lorenz attractor.
    This numerically integrates the Lorenz system and draws the trajectory.
    """

    def lorenz(self, x, y, z, s=10, r=28, b=8/3):
        """ Lorenz system differential equations. """
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        return dx, dy, dz

    def generate_lorenz_points(self, n_points=5000, dt=0.005):
        """ Numerically integrate the Lorenz system with Euler’s method. """
        x, y, z = 1.0, 1.0, 1.0   # initial condition
        points = []

        for _ in range(n_points):
            dx, dy, dz = self.lorenz(x, y, z)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            points.append([x, y, z])

        return np.array(points)

    def construct(self):
        # Setup camera
        self.set_camera_orientation(phi=180 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.05)       

        pts = self.generate_lorenz_points()

        # Convert points to Manim space (scale for visibility)
        scaled_points = pts * 0.08
        path = VMobject(color=BLUE, stroke_width=0.8).set_points_smoothly(scaled_points)

        # Animation: draw the attractor
        self.play(Create(path), run_time=17, rate_func=linear)
        self.wait()

        self.play(FadeOut(path))
        self.wait(2)

class DoublePendulum(ThreeDScene):
    def construct(self):
        # Physical parameters
        g = 9.81  # gravity
        L1 = 1.0  # length of pendulum 1
        L2 = 1.0  # length of pendulum 2
        m1 = 1.0  # mass of pendulum 1
        m2 = 1.0  # mass of pendulum 2

        self.camera.background_color = "#0a0a0a"
        colour1 = "#39FF14" # Neon green
        colour2 = "#00FFFF"  # Cyan
        
        # Initial conditions [theta1, omega1, theta2, omega2]
        # Two pendulums with very slightly different initial conditions
        initial1 = [np.pi/2, 0, np.pi/2, 0]  # First pendulum
        initial2 = [np.pi/2 + 0.05, 0, np.pi/2, 0]  # Second pendulum (tiny difference)
        
        # Time parameters
        # FAST TEST: dt=0.1, t_max=5 (renders in ~30 seconds)
        # NORMAL: dt=0.03, t_max=20 (renders in ~3-5 minutes)
        dt = 0.08  # time step for animation (larger = faster but choppier)
        t_max = 20 # total simulation time (shorter = faster)
        time_points = np.arange(0, t_max, dt)
        
        # Equations of motion for double pendulum
        def derivatives(state, t, L1, L2, m1, m2, g):
            theta1, omega1, theta2, omega2 = state
            
            delta = theta2 - theta1
            den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
            den2 = (L2 / L1) * den1
            
            dydt = np.zeros_like(state)
            dydt[0] = omega1
            
            dydt[1] = (m2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta)
                      + m2 * g * np.sin(theta2) * np.cos(delta)
                      + m2 * L2 * omega2 * omega2 * np.sin(delta)
                      - (m1 + m2) * g * np.sin(theta1)) / den1
            
            dydt[2] = omega2
            
            dydt[3] = (-m2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta)
                      + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
                      - (m1 + m2) * L1 * omega1 * omega1 * np.sin(delta)
                      - (m1 + m2) * g * np.sin(theta2)) / den2
            
            return dydt
        
        # Solve the ODEs
        solution1 = odeint(derivatives, initial1, time_points, args=(L1, L2, m1, m2, g))
        solution2 = odeint(derivatives, initial2, time_points, args=(L1, L2, m1, m2, g))
        
        # Scale factor for display
        scale = 0.5

        # Create pivot point
        pivot = Dot(ORIGIN, color=WHITE, radius=0.008)
        self.play(Create(pivot))

        stroke_width = 1
        trail_width = 0.8
        
        # Create pendulum objects for both systems
        # Pendulum 1 (green)
        bob1_1 = Dot(color=colour1, radius=0.012)
        bob2_1 = Dot(color=colour2, radius=0.012)
        rod1_1 = Line(ORIGIN, ORIGIN, color=colour1, stroke_width=stroke_width)
        rod2_1 = Line(ORIGIN, ORIGIN, color=colour1, stroke_width=stroke_width)
        
        # Pendulum 2 (cyan)
        bob1_2 = Dot(color=colour2, radius=0.012)
        bob2_2 = Dot(color=colour2, radius=0.012)
        rod1_2 = Line(ORIGIN, ORIGIN, color=colour2, stroke_width=stroke_width)
        rod2_2 = Line(ORIGIN, ORIGIN, color=colour2, stroke_width=stroke_width)
        
        # Trails for the second bob of each pendulum
        trail1 = VMobject(color=colour1, stroke_width=trail_width)
        trail2 = VMobject(color=colour2, stroke_width=trail_width)
        trail1.set_points_as_corners([ORIGIN, ORIGIN])
        trail2.set_points_as_corners([ORIGIN, ORIGIN])
        
        self.add(rod1_1, rod2_1, bob1_1, bob2_1)
        self.add(rod1_2, rod2_2, bob1_2, bob2_2)
        self.add(trail1, trail2)
        
        # Animation loop
        trail1_points = [ORIGIN]
        trail2_points = [ORIGIN]
        
        for i in range(len(time_points)):
            # Get angles for both pendulums
            theta1_1, _, theta2_1, _ = solution1[i]
            theta1_2, _, theta2_2, _ = solution2[i]
            
            # Calculate positions for pendulum 1 
            x1_1 = scale * L1 * np.sin(theta1_1)
            y1_1 = -scale * L1 * np.cos(theta1_1)
            pos1_1 = np.array([x1_1, y1_1, 0])
            
            x2_1 = x1_1 + scale * L2 * np.sin(theta2_1)
            y2_1 = y1_1 - scale * L2 * np.cos(theta2_1)
            pos2_1 = np.array([x2_1, y2_1, 0])
            
            # Calculate positions for pendulum 2
            x1_2 = scale * L1 * np.sin(theta1_2)
            y1_2 = -scale * L1 * np.cos(theta1_2)
            pos1_2 = np.array([x1_2, y1_2, 0])
            
            x2_2 = x1_2 + scale * L2 * np.sin(theta2_2)
            y2_2 = y1_2 - scale * L2 * np.cos(theta2_2)
            pos2_2 = np.array([x2_2, y2_2, 0])
            
            # Update pendulum 1
            rod1_1.put_start_and_end_on(ORIGIN, pos1_1)
            rod2_1.put_start_and_end_on(pos1_1, pos2_1)
            bob1_1.move_to(pos1_1)
            bob2_1.move_to(pos2_1)
            
            # Update pendulum 2
            rod1_2.put_start_and_end_on(ORIGIN, pos1_2)
            rod2_2.put_start_and_end_on(pos1_2, pos2_2)
            bob1_2.move_to(pos1_2)
            bob2_2.move_to(pos2_2)
            
            # Update trails (only keep last 100 points for performance)
            # For even faster testing, reduce to 30-50 points
            trail1_points.append(pos2_1)
            trail2_points.append(pos2_2)
            if len(trail1_points) > 100: 
                trail1_points.pop(0)
                trail2_points.pop(0)
            
            trail1.set_points_as_corners(trail1_points)
            trail2.set_points_as_corners(trail2_points)
            
            self.wait(dt)


class RosslerAttractor(ThreeDScene):
    def construct(self):
        # Set camera
        self.set_camera_orientation(phi=120*DEGREES, theta=100*DEGREES, zoom=0.2)
        self.begin_ambient_camera_rotation(rate=0.2) 
        
        # Define Rössler system
        def rossler(t, state, a=0.2, b=0.2, c=5.7):
            x, y, z = state
            dx = -y - z
            dy = x + a*y
            dz = b + z*(x - c)
            return [dx, dy, dz]

        # Integrate system
        t_span = (0, 200)
        t_eval = np.linspace(*t_span, 10000)
        initial_state = [0.0, 0.0, 0.0]
        sol = solve_ivp(rossler, t_span, initial_state, t_eval=t_eval)

        # Scale points for Manim
        points = np.array([sol.y[0], sol.y[1], sol.y[2]]).T
        points *= 0.3  # scale down for visibility

        # Create 3D trajectory as VMobject
        trajectory = VMobject(color=BLUE, stroke_width=0.5)
        trajectory.set_points_smoothly(points)

        # Animate trajectory being drawn
        self.play(Create(trajectory), run_time=40, rate_func=linear)
        self.wait(2)


class MbZoom(Scene):
    def mandelbrot_array(self, x_min, x_max, y_min, y_max, width, height, max_iter=200):
        """
        Compute a Mandelbrot set and return a 2D numpy array.
        Values normalized 0-1.
        """
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        C = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        Z = np.zeros_like(C)
        div_time = np.zeros(C.shape, dtype=int)

        for i in range(max_iter):
            Z = Z**2 + C
            diverged = np.abs(Z) > 2
            div_now = diverged & (div_time == 0)
            div_time[div_now] = i
            Z[diverged] = 2

        return div_time / max_iter  # normalize

    def construct(self):
        # 1️⃣ Initial Mandelbrot image
        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.5, 1.5
        width, height = 800, 600

        mandelbrot_data = self.mandelbrot_array(x_min, x_max, y_min, y_max, width, height)
        mandelbrot_img = ImageMobject((mandelbrot_data * 255).astype(np.uint8))
        mandelbrot_img.scale(3)  # scale to fit scene
        self.add(mandelbrot_img)
        self.wait(1)

        # 2️⃣ Define a list of zoom centers and zoom factors
        zooms = [
            (-0.7435 + 0.1314j, 0.05),   # Seahorse Valley
            (-0.7436438870371587 + 0.13182590420531197j, 0.01),  # deeper zoom
            (-0.7436438870371587047 + 0.13182590420531197049j, 0.002)  # extreme zoom
        ]

        for center, zoom_factor in zooms:
            # Calculate target position in Manim coordinates
            target_pos = np.array([-center.real * 10, -center.imag * 10, 0])
            # Animate scaling and moving
            self.play(
                mandelbrot_img.animate.scale(1/zoom_factor).move_to(target_pos),
                run_time=5,
                rate_func=smooth
            )
            self.wait(0.5)

        # Optional final hold
        self.wait(2)
