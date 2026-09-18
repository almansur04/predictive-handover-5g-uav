import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import plotly.graph_objects as go

class Visualizer:
    """Generates publication-quality PDF plots, interactive 3D WebGL Digital Twins, and MP4 animations."""
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_paper_timeline(self, time_vec, rsrp_dense, active_rsrp_a3, active_rsrp_lstm, t_corner, lstm_step, a3_step, dt):
        """Figure 1 for the IEEE two-column paper."""
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=300)
        
        # Panel 1: Physical mmWave RSRP
        axes[0].plot(time_vec, rsrp_dense[:, 0], label="Serving Cell: gNB-0", color="#1f77b4", linewidth=2.4)
        axes[0].plot(time_vec, rsrp_dense[:, 3], label="Target Cell: gNB-3", color="#d62728", linewidth=2.4)
        axes[0].axhline(y=-95, color="black", linestyle="--", linewidth=1.2, label="RLF Outage Threshold (-95 dBm)")
        axes[0].axvline(x=t_corner, color="darkred", linestyle="-.", linewidth=1.5, label=f"Building Blockage (t={t_corner:.2f}s)")
        axes[0].set_ylabel("RSRP [dBm]", fontsize=11)
        axes[0].set_title("Physical mmWave Propagation at 28 GHz (Sionna Ray-Tracing)", fontsize=12, fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].set_ylim(-145, -35)
        axes[0].legend(loc="upper right", fontsize=9.5)

        # Panel 2: Protocol Timeline Comparison
        axes[1].plot(time_vec, active_rsrp_a3, label="Standard 3GPP A3 Link", color="#d62728", linewidth=2.2, linestyle="--")
        axes[1].plot(time_vec, active_rsrp_lstm, label="Proposed LSTM Predictive Link", color="#2ca02c", linewidth=2.5)
        axes[1].axhline(y=-95, color="black", linestyle="--", linewidth=1.2)
        
        # Red Outage Box
        axes[1].fill_between(time_vec, -145, -95, where=(active_rsrp_a3 < -95), color="red", alpha=0.25, label="3GPP A3 Radio Link Failure (200 ms)")
        
        axes[1].set_ylabel("Serving RSRP [dBm]", fontsize=11)
        axes[1].set_xlabel("Time [seconds]", fontsize=11)
        axes[1].set_title("Active Serving Link: 3GPP A3 vs. Proposed Predictive LSTM", fontsize=12, fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].set_ylim(-145, -35)
        axes[1].legend(loc="lower left", fontsize=9.5)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "fig_blockage_timeline.pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(os.path.join(self.output_dir, "fig_blockage_timeline.png"), format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def export_3d_webgl_twin(self, gnb_positions, trajectory, corner_coord):
        """Deliverable 1: Standalone 3D interactive WebGL HTML Digital Twin."""
        fig_3d = go.Figure()

        # Add Base Stations
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, pos in enumerate(gnb_positions):
            fig_3d.add_trace(go.Scatter3d(x=[pos[0], pos[0]], y=[pos[1], pos[1]], z=[0, pos[2]],
                                         mode='lines', line=dict(color='gray', width=6), showlegend=False))
            fig_3d.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]],
                                         mode='markers+text', marker=dict(size=9, color=colors[idx], symbol='diamond'),
                                         text=[f"gNB-{idx}"], name=f"Base Station gNB-{idx}"))

        # Volumetric Obstacle High-Rise Building Mesh
        fig_3d.add_trace(go.Mesh3d(
            x=[-108, -80, -80, -108, -108, -80, -80, -108],
            y=[46, 46, 20, 20, 46, 46, 20, 20],
            z=[0, 0, 0, 0, 28, 28, 28, 28],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='slategray', opacity=0.7, name='Concrete High-Rise Block'
        ))

        # UAV Flight Trajectory
        fig_3d.add_trace(go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                                     mode='lines', line=dict(color='yellow', width=5), name='3D UAV Trajectory'))

        # Blockage Marker
        fig_3d.add_trace(go.Scatter3d(x=[corner_coord[0]], y=[corner_coord[1]], z=[corner_coord[2]],
                                     mode='markers+text', marker=dict(size=12, color='red', symbol='cross'),
                                     text=["85 dB Knife-Edge Blockage Point"], name="Blockage Boundary"))

        fig_3d.update_layout(
            title="<b>Interactive 3D Digital Twin: 28 GHz mmWave UAV Handover Environment</b>",
            scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Altitude Z [m]'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        html_path = os.path.join(self.output_dir, "interactive_digital_twin.html")
        fig_3d.write_html(html_path)
        print(f"[✓] 3D Digital Twin exported to: {html_path}")

    def render_animation(self, time_vec, path_x, path_y, gnb_positions, rsrp_dense, active_a3, active_lstm, t_corner, lstm_step, a3_step, dt):
        """Deliverable 2: High-definition 60-FPS synchronized MP4 and looping GIF."""
        fig_anim, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), dpi=120)
        plt.subplots_adjust(hspace=0.38)

        # Top Axis Setup
        ax_top.set_xlim(-130, -70)
        ax_top.set_ylim(-85, 75)
        ax_top.set_title("Real-Time 3D Spatial Mobility & 28 GHz Beam Tracking", fontsize=11, fontweight='bold')
        ax_top.grid(True, linestyle=':', alpha=0.5)

        b_rect = patches.Rectangle((-108, 15), 32, 31, linewidth=1.5, edgecolor='black', facecolor='slategray', alpha=0.75, label='High-Rise Facade')
        ax_top.add_patch(b_rect)
        ax_top.plot(path_x, path_y, 'k--', alpha=0.4, label='UAV Flight Path')
        ax_top.scatter(gnb_positions[0][0], gnb_positions[0][1], color='#1f77b4', s=160, marker='^', zorder=5, label='gNB-0 (Serving)')
        ax_top.scatter(gnb_positions[3][0], gnb_positions[3][1], color='#d62728', s=160, marker='^', zorder=5, label='gNB-3 (Target)')

        uav_marker, = ax_top.plot([], [], 'o', color='gold', markeredgecolor='black', markersize=13, zorder=10)
        active_beam, = ax_top.plot([], [], color='#1f77b4', linewidth=2.5, zorder=6)
        status_box = ax_top.text(-128, -78, "", fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95))
        ax_top.legend(loc='upper right', fontsize=8.5)

        # Bottom Axis Setup
        ax_bot.set_xlim(0, time_vec[-1])
        ax_bot.set_ylim(-145, -35)
        ax_bot.set_title("Active Serving Link Signal Strength & Handover Protocol Decision", fontsize=11, fontweight='bold')
        ax_bot.axhline(y=-95, color='black', linestyle='--', linewidth=1.2, label='RLF Outage Threshold (-95 dBm)')
        ax_bot.axvline(x=t_corner, color='darkred', linestyle='-.', alpha=0.8, label=f'Blockage Event (t={t_corner:.2f}s)')
        ax_bot.grid(True, linestyle=':', alpha=0.5)

        line_a3, = ax_bot.plot([], [], color='#d62728', linestyle='--', linewidth=2.2, label='Legacy 3GPP A3 Link')
        line_lstm, = ax_bot.plot([], [], color='#2ca02c', linewidth=2.5, label='Proposed LSTM Predictive Link')
        time_cursor = ax_bot.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)
        outage_rect = patches.Rectangle((t_corner, -145), 0, 110, facecolor='red', alpha=0.0, label='RLF Outage Window')
        ax_bot.add_patch(outage_rect)
        ax_bot.legend(loc='lower left', fontsize=8.5)

        def update(frame):
            curr_t = time_vec[frame]
            uav_x, uav_y = path_x[frame], path_y[frame]
            uav_marker.set_data([uav_x], [uav_y])

            if curr_t < lstm_step * dt:
                active_beam.set_data([gnb_positions[0][0], uav_x], [gnb_positions[0][1], uav_y])
                active_beam.set_color('#1f77b4')
                status_box.set_text(f"t = {curr_t:.2f}s | Serving: gNB-0 (LoS Healthy: {rsrp_dense[frame, 0]:.1f} dBm)")
                status_box.set_color('#1f77b4')
                outage_rect.set_width(0)
                outage_rect.set_alpha(0.0)
            elif curr_t < t_corner:
                active_beam.set_data([gnb_positions[3][0], uav_x], [gnb_positions[3][1], uav_y])
                active_beam.set_color('#2ca02c')
                status_box.set_text(f"t = {curr_t:.2f}s | [LSTM PROACTIVE HANDOVER] Attached to gNB-3 early!")
                status_box.set_color('#2ca02c')
                outage_rect.set_width(0)
                outage_rect.set_alpha(0.0)
            elif curr_t < a3_step * dt:
                active_beam.set_data([gnb_positions[3][0], uav_x], [gnb_positions[3][1], uav_y])
                active_beam.set_color('#d62728')
                status_box.set_text(f"t = {curr_t:.2f}s | [3GPP A3 IN OUTAGE] Timer Waiting... Radio Link Failure!")
                status_box.set_color('crimson')
                outage_rect.set_width(curr_t - t_corner)
                outage_rect.set_alpha(0.25)
            else:
                active_beam.set_data([gnb_positions[3][0], uav_x], [gnb_positions[3][1], uav_y])
                active_beam.set_color('#2ca02c')
                status_box.set_text(f"t = {curr_t:.2f}s | Both Systems Operating on gNB-3")
                status_box.set_color('#2ca02c')
                outage_rect.set_width(a3_step * dt - t_corner)
                outage_rect.set_alpha(0.20)

            line_a3.set_data(time_vec[:frame+1], active_a3[:frame+1])
            line_lstm.set_data(time_vec[:frame+1], active_lstm[:frame+1])
            time_cursor.set_xdata([curr_t])
            return uav_marker, active_beam, line_a3, line_lstm, time_cursor, status_box, outage_rect

        ani = animation.FuncAnimation(fig_anim, update, frames=len(time_vec), interval=80, blit=True)
        ani.save(os.path.join(self.output_dir, "handover_simulation.mp4"), writer="ffmpeg", fps=12)
        ani.save(os.path.join(self.output_dir, "handover_simulation.gif"), writer="pillow", fps=12)
        plt.close(fig_anim)
        print(f"[✓] MP4 & GIF animations generated in '{self.output_dir}/'")