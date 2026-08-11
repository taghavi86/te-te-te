"""
Streamlit Dashboard for Tennis AI Coach
Professional UI with real-time visualization and analysis
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import pandas as pd

def render_dashboard(results: Dict, analysis_options: List[str]):
    """Main dashboard rendering function"""
    
    # Create tabs for different views
    tabs = st.tabs([
        "📊 Overview", 
        "🎾 Technique Analysis", 
        "📈 Trajectories", 
        "⚖️ Pro Comparison",
        "💡 Recommendations"
    ])
    
    with tabs[0]:
        render_overview_tab(results)
    
    with tabs[1]:
        render_technique_tab(results)
    
    with tabs[2]:
        render_trajectory_tab(results, analysis_options)
    
    with tabs[3]:
        render_comparison_tab(results)
    
    with tabs[4]:
        render_recommendations_tab(results)


def render_overview_tab(results: Dict):
    """Render overview statistics and metrics"""
    
    st.header("📊 Analysis Overview")
    
    # Metadata
    metadata = results.get('metadata', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Video Duration",
            f"{metadata.get('duration', 0):.1f}s",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Frames",
            f"{metadata.get('total_frames', 0)}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Frame Rate",
            f"{metadata.get('fps', 0):.0f} FPS",
            delta=None
        )
    
    with col4:
        resolution = metadata.get('resolution', (0, 0))
        st.metric(
            "Resolution",
            f"{resolution[0]}x{resolution[1]}",
            delta=None
        )
    
    # Overall statistics
    st.subheader("🎯 Performance Metrics")
    
    statistics = results.get('statistics', {})
    
    cols = st.columns(3)
    
    with cols[0]:
        if 'technique' in statistics:
            tech_stats = statistics['technique']
            avg_score = tech_stats.get('average_score', 0)
            consistency = tech_stats.get('consistency', 0)
            
            st.metric(
                "Average Technique Score",
                f"{avg_score:.1f}/100",
                delta=f"{consistency*100:.0f}% consistency"
            )
    
    with cols[1]:
        if 'ball' in statistics:
            ball_stats = statistics['ball']
            max_speed = ball_stats.get('max_speed', 0)
            
            st.metric(
                "Max Ball Speed",
                f"{max_speed*10:.1f} km/h",
                delta=None
            )
    
    with cols[2]:
        if 'pose' in statistics:
            pose_stats = statistics['pose']
            
            st.metric(
                "Movement Consistency",
                f"{(1 - np.mean(pose_stats.get('angle_variance', [1])))*100:.1f}%",
                delta=None
            )
    
    # Frame-by-frame score chart
    st.subheader("📈 Technique Score Over Time")
    
    frames = results.get('frames', [])
    if frames:
        scores = []
        timestamps = []
        
        for frame in frames:
            if 'technique' in frame:
                scores.append(frame['technique'].get('score', 0))
                timestamps.append(frame.get('timestamp', 0))
        
        if scores:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=scores,
                mode='lines+markers',
                name='Technique Score',
                line=dict(color='#00CC96', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Time (s)",
                yaxis_title="Score",
                yaxis_range=[0, 100],
                hovermode='x unified',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_technique_tab(results: Dict):
    """Render detailed technique analysis"""
    
    st.header("🎾 Technique Analysis")
    
    frames = results.get('frames', [])
    
    if not frames:
        st.warning("No technique data available")
        return
    
    # Extract technique data
    technique_data = []
    for i, frame in enumerate(frames):
        if 'technique' in frame:
            tech = frame['technique']
            technique_data.append({
                'Frame': i,
                'Score': tech.get('score', 0),
                'Phase': tech.get('phase', 'unknown'),
                'Stance': tech.get('component_scores', {}).get('stance', 0),
                'Backswing': tech.get('component_scores', {}).get('backswing', 0),
                'Contact': tech.get('component_scores', {}).get('contact_point', 0),
                'Follow-through': tech.get('component_scores', {}).get('follow_through', 0),
                'Balance': tech.get('component_scores', {}).get('balance', 0)
            })
    
    if technique_data:
        df = pd.DataFrame(technique_data)
        
        # Component scores radar chart
        st.subheader("🕸️ Technique Components")
        
        components = ['Stance', 'Backswing', 'Contact', 'Follow-through', 'Balance']
        avg_components = [df[col].mean() for col in components]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=avg_components,
            theta=components,
            fill='toself',
            name='Average Performance',
            line_color='#636EFA'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=500,
            template='plotly_dark'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Component Statistics")
            
            stats_df = pd.DataFrame({
                'Component': components,
                'Average': avg_components,
                'Min': [df[col].min() for col in components],
                'Max': [df[col].max() for col in components],
                'Std Dev': [df[col].std() for col in components]
            })
            
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # Phase distribution
        st.subheader("🔄 Stroke Phase Distribution")
        
        phase_counts = df['Phase'].value_counts()
        
        fig = px.pie(
            values=phase_counts.values,
            names=phase_counts.index,
            title='Time Spent in Each Phase',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        
        fig.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)


def render_trajectory_tab(results: Dict, analysis_options: List[str]):
    """Render ball and racket trajectories"""
    
    st.header("📈 Trajectory Analysis")
    
    trajectories = results.get('trajectories', {})
    
    # Ball trajectory
    if 'Ball Tracking' in analysis_options and 'ball' in trajectories:
        st.subheader("🎾 Ball Trajectory")
        
        ball_traj = trajectories['ball']
        
        if ball_traj:
            ball_array = np.array(ball_traj)
            
            # 2D trajectory plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ball_array[:, 0],
                y=ball_array[:, 1],
                mode='markers+lines',
                name='Ball Path',
                marker=dict(
                    size=8,
                    color=ball_array[:, 1],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Height")
                ),
                line=dict(width=3, color='#FFA15A')
            ))
            
            fig.update_layout(
                height=500,
                xaxis_title="X Position",
                yaxis_title="Y Position",
                template='plotly_dark',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trajectory statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if len(ball_array) > 1:
                    speed = np.max(np.linalg.norm(np.diff(ball_array, axis=0), axis=1))
                    st.metric("Max Speed", f"{speed*10:.1f} units/s")
            
            with col2:
                coverage = np.ptp(ball_array[:, 0]) * np.ptp(ball_array[:, 1])
                st.metric("Coverage Area", f"{coverage:.0f} sq units")
            
            with col3:
                st.metric("Trajectory Points", len(ball_array))
    
    # Racket trajectory
    if 'Racket Detection' in analysis_options and 'racket' in trajectories:
        st.subheader("🏓 Racket Trajectory")
        
        racket_traj = trajectories['racket']
        
        if racket_traj:
            racket_array = np.array(racket_traj)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=racket_array[:, 0],
                y=racket_array[:, 1],
                mode='markers+lines',
                name='Racket Path',
                marker=dict(size=10, color='#EF553B'),
                line=dict(width=3, color='#EF553B', dash='dash')
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="X Position",
                yaxis_title="Y Position",
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Joint trajectories
    if 'Pose Detection' in analysis_options and 'joints' in trajectories:
        st.subheader("🦴 Joint Movement Patterns")
        
        joints = trajectories['joints']
        
        if joints:
            # Show key joint angles over time
            joint_angles = []
            timestamps = []
            
            for i, joint_frame in enumerate(joints):
                if isinstance(joint_frame, list) and len(joint_frame) > 0:
                    joint_angles.append(joint_frame[:4])  # First 4 angles
                    timestamps.append(i / 30)  # Assuming 30 FPS
            
            if joint_angles:
                joint_array = np.array(joint_angles)
                
                fig = go.Figure()
                
                joint_names = ['Elbow', 'Knee', 'Shoulder Rotation', 'Hip-Knee']
                colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
                
                for i, name in enumerate(joint_names):
                    if i < joint_array.shape[1]:
                        fig.add_trace(go.Scatter(
                            x=timestamps,
                            y=joint_array[:, i],
                            mode='lines',
                            name=name,
                            line=dict(color=colors[i], width=2)
                        ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Time (s)",
                    yaxis_title="Angle (degrees)",
                    template='plotly_dark',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)


def render_comparison_tab(results: Dict):
    """Render professional comparison results"""
    
    st.header("⚖️ Professional Comparison")
    
    if 'comparison' not in results:
        st.info("Upload a professional reference video to enable comparison")
        return
    
    comparison = results['comparison']
    
    # Overall similarity
    overall_sim = comparison.get('overall_similarity', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Overall Similarity",
            f"{overall_sim*100:.1f}%",
            delta=None
        )
    
    with col2:
        best_pro = comparison.get('best_matching_pro', 'Unknown')
        st.metric("Best Match", best_pro)
    
    # Detailed scores
    st.subheader("📊 Detailed Comparison")
    
    detailed_scores = comparison.get('detailed_scores', {})
    
    if detailed_scores:
        df = pd.DataFrame({
            'Component': list(detailed_scores.keys()),
            'Similarity': [v*100 for v in detailed_scores.values()]
        })
        
        fig = px.bar(
            df,
            x='Component',
            y='Similarity',
            title='Component-wise Similarity',
            color='Similarity',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            yaxis_title="Similarity (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    temporal = comparison.get('temporal_analysis', {})
    
    if temporal:
        st.subheader("⏱️ Timing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            timing_score = temporal.get('timing_score', 0)
            st.metric("Timing Score", f"{timing_score*100:.1f}%")
        
        with col2:
            rhythm = temporal.get('rhythm_consistency', 0)
            st.metric("Rhythm Consistency", f"{rhythm*100:.1f}%")
        
        phase_diffs = temporal.get('phase_differences', [])
        
        if phase_diffs:
            phase_df = pd.DataFrame(phase_diffs)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='User Timing',
                x=phase_df['phase'],
                y=phase_df['user_timing'],
                marker_color='#636EFA'
            ))
            
            fig.add_trace(go.Bar(
                name='Pro Timing',
                x=phase_df['phase'],
                y=phase_df['pro_timing'],
                marker_color='#00CC96'
            ))
            
            fig.update_layout(
                barmode='group',
                height=400,
                xaxis_title="Phase",
                yaxis_title="Normalized Time",
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_recommendations_tab(results: Dict):
    """Render coaching recommendations"""
    
    st.header("💡 AI Coach Recommendations")
    
    frames = results.get('frames', [])
    
    all_recommendations = []
    all_feedback = []
    
    for frame in frames:
        if 'technique' in frame:
            tech = frame['technique']
            
            recs = tech.get('recommendations', [])
            feedback = tech.get('feedback', [])
            
            all_recommendations.extend(recs)
            all_feedback.extend(feedback)
    
    # Most common recommendations
    if all_recommendations:
        st.subheader("🎯 Priority Recommendations")
        
        from collections import Counter
        rec_counts = Counter(all_recommendations)
        
        for i, (rec, count) in enumerate(rec_counts.most_common(5), 1):
            with st.expander(f"Recommendation {i} ({count} occurrences)", expanded=(i==1)):
                st.write(rec)
                
                # Add drill suggestions
                if "stance" in rec.lower():
                    st.info("💪 Drill: Practice shadow swings with exaggerated wide stance")
                elif "backswing" in rec.lower():
                    st.info("💪 Drill: Use mirror work to check shoulder rotation")
                elif "contact" in rec.lower():
                    st.info("💪 Drill: Ball toss practice for consistent contact height")
                elif "follow" in rec.lower():
                    st.info("💪 Drill: Exaggerated follow-through exercises")
                elif "balance" in rec.lower():
                    st.info("💪 Drill: Single-leg balance exercises")
    
    # Feedback summary
    if all_feedback:
        st.subheader("📝 Technique Feedback")
        
        feedback_counts = Counter(all_feedback)
        
        for feedback, count in feedback_counts.most_common(5):
            st.warning(f"⚠️ {feedback} ({count} times)")
    
    # Pro comparison recommendations
    if 'comparison' in results:
        comparison = results['comparison']
        pro_recs = comparison.get('recommendations', [])
        
        if pro_recs:
            st.subheader("🏆 Professional-Level Tips")
            
            for rec in pro_recs:
                st.success(f"✅ {rec}")
    
    # Generate training plan
    st.subheader("📅 Suggested Training Plan")
    
    if all_recommendations:
        rec_counts = Counter(all_recommendations)
        top_issues = [item[0] for item in rec_counts.most_common(3)]
        
        training_plan = {
            "Week 1-2": f"Focus on: {top_issues[0] if len(top_issues) > 0 else 'Fundamentals'}",
            "Week 3-4": f"Focus on: {top_issues[1] if len(top_issues) > 1 else 'Consistency'}",
            "Week 5-6": f"Focus on: {top_issues[2] if len(top_issues) > 2 else 'Advanced technique'}",
            "Ongoing": "Video analysis after each practice session"
        }
        
        for period, focus in training_plan.items():
            st.info(f"**{period}**: {focus}")
