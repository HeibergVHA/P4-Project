import control
import numpy as np
import pandas as pd

s = control.tf('s')

# ── Physical constants ─────────────────────────────────────────────────────
g      = 9.81   # m/s²
J      = 0.02167 # 0.02166666666666667   # moment of inertia (kg·m²) — tune to your x500 airframe
                # x500 typical: Ixx ≈ Iyy ≈ 0.01–0.02 kg·m²

# ── PX4 gains ─────────────────────────────────────────────────────────────
Kp_pos  = 0.95
Kp_vel  = 1.8;   Ki_vel  = 0.4;   Kd_vel  = 0.2
Kp_att  = 4.0
Kp_rate = 0.15;  Ki_rate = 0.2;   Kd_rate = 0.003

# ══════════════════════════════════════════════════════════════════════════
# PLANT CHAIN  (innermost → outermost)
#
#   τ_cmd ──[1/J·s]──► θ_dot ──[1/s]──► θ ──[g/s]──► ẋ ──[1/s]──► x
#
#   τ_cmd  : torque command          (N·m)
#   θ_dot  : pitch rate              (rad/s)
#   θ      : pitch angle             (rad)
#   ẋ      : horizontal velocity     (m/s)
#   x      : horizontal position     (m)
# ══════════════════════════════════════════════════════════════════════════

P_angacc  = control.tf([1],    [1, 0])      # τ → θ_dot     (1/Js)
P_att     = control.tf([1],    [1, 0])      # θ_dot → θ     (1/s)
P_vel     = control.tf([g],    [1, 0])      # θ → ẋ         (g/s)
P_pos     = control.tf([1],    [1, 0])      # ẋ → x         (1/s)

# ══════════════════════════════════════════════════════════════════════════
# CONTROLLERS
#
#   Loop 4 (innermost): rate PID   τ_cmd   = C_rate · (θ_dot_ref − θ_dot)
#   Loop 3:             att  P     θ_dot_ref = C_att  · (θ_ref    − θ)
#   Loop 2:             vel  PID   θ_ref     = C_acc  · C_vel · (ẋ_ref − ẋ)
#   Loop 1 (outermost): pos  P     ẋ_ref     = C_pos  · (x_ref  − x)
# ══════════════════════════════════════════════════════════════════════════

C_rate = control.tf([Kd_rate, Kp_rate, Ki_rate], [1, 0])   # PID on pitch rate
C_att  = Kp_att                                              # P   on pitch angle
C_vel  = control.tf([Kd_vel,  Kp_vel,  Ki_vel],  [1, 0])   # PID on velocity
C_acc  = 1 / g                                               # accel → angle conversion
C_pos  = Kp_pos                                              # P   on position

# ══════════════════════════════════════════════════════════════════════════
# CLOSE LOOPS — inside out
# ══════════════════════════════════════════════════════════════════════════

# Loop 4: rate loop   θ_dot_ref → θ_dot
#   open-loop: C_rate · P_angacc
#   C_rate acts on rate error; plant is τ → θ_dot (= 1/Js)
G_rate = control.feedback(C_rate * P_angacc* 1/J, 1)

# Loop 3: attitude loop   θ_ref → θ
#   open-loop: C_att · (1/s) · G_rate
#   C_att produces a rate reference; G_rate tracks it; integrate rate → angle
G_att  = control.feedback(C_att * P_att * G_rate, 1)

# Loop 2: velocity loop   ẋ_ref → ẋ
#   open-loop: C_vel · C_acc · P_vel · G_att
#   C_vel·C_acc maps velocity error to an angle reference
G_vel  = control.feedback(C_vel * C_acc * P_vel * G_att, 1)

# Loop 1: position loop   x_ref → x
#   open-loop: C_pos · P_pos · G_vel
#   C_pos maps position error to a velocity reference
G_pos  = control.feedback(C_pos * P_pos * G_vel, 1)

# ── Minimal realisation & display ─────────────────────────────────────────
G_min = control.minreal(G_pos, verbose=False)
print("Full closed-loop TF  (x_ref → x):")
print(G_min)

poles = control.poles(G_pos)
print("\nClosed-loop poles:")
for p in sorted(poles, key=lambda z: z.real):
    tag = "stable" if p.real < 0 else "*** UNSTABLE ***"
    print(f"  {p:+.4f}   {tag}")

# -- Yes
# Load CSV
df = pd.read_csv("step_1_avg/data.csv")

data1 = df['/fmu/out/vehicle_odometry/position[0]'].values[1525:2525]
data2 = 1-df['/fmu/out/vehicle_odometry/position[0]'].values[525:1525]

# ── Step response ─────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
t_out, y_out = control.step_response(G_pos, T=t)

plt.figure(figsize=(9, 4))
plt.plot(t_out, y_out)
plt.plot(t, data1)
plt.plot(t, data2)
plt.axhline(1.0, color='k', linestyle='--', linewidth=0.8, label='Setpoint')
plt.title('Step Response of X-axis position (using PX4-based script)')
plt.xlabel('Time (s)')
plt.ylabel('X-position (m)')
plt.legend(['Transfer function', 'Px4-Autopilot 0m->1m (#1)', 'Px4-Autopilot 1m->0m (#1)'])
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()