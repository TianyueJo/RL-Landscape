"""
Visualize GridWorld environment configuration (text + HTML)
"""
import argparse
from env import GridSpec, make_env


def visualize_gridworld_text(spec: GridSpec):
    """
    Visualize a GridWorld spec using an ASCII grid.
    
    Args:
        spec: GridSpec configuration
    """
    width = spec.width
    height = spec.height
    
    # Create grid
    grid = [["." for _ in range(width)] for _ in range(height)]
    
    # Mark walls
    for wx, wy in spec.walls:
        grid[wy][wx] = "#"
    
    # Mark start
    sx, sy = spec.start
    grid[sy][sx] = "S"
    
    # Mark goal
    gx, gy = spec.goal
    grid[gy][gx] = "G"
    
    # Mark candy cells
    candies = getattr(spec, "candies", ()) or (() if spec.candy is None else (spec.candy,))
    for cx, cy in candies:
        grid[cy][cx] = "C"
    
    # Mark risk cell
    if spec.risk is not None:
        rx, ry = spec.risk
        grid[ry][rx] = "R"
    
    # Print grid (top-to-bottom; y increases downward)
    print("=" * 60)
    print(f"GridWorld Environment ({width}×{height})")
    print("=" * 60)
    print()
    print("Legend:")
    print("  S = Start")
    print("  G = Goal")
    print("  C = Candy")
    print("  R = Risk")
    print("  # = Wall")
    print("  . = Empty")
    print()
    print("Grid Layout (coordinate system: (x, y), top-left is (0, 0)):")
    print()
    
    # Print column indices
    print("   ", end="")
    for x in range(width):
        print(f"{x} ", end="")
    print()
    
    # Print grid rows
    for y in range(height):
        print(f"{y:2} ", end="")
        for x in range(width):
            print(f"{grid[y][x]} ", end="")
        print()
    
    print()
    print("=" * 60)
    print("Environment Configuration:")
    print("=" * 60)
    print(f"  Start Position: {spec.start}")
    print(f"  Goal Position: {spec.goal}")
    if candies:
        print(f"  Candy Position(s): {list(candies)}")
    if spec.risk is not None:
        print(f"  Risk Position: {spec.risk} (failure probability: {spec.risk_p})")
    print(f"  Goal Reward: {spec.goal_reward}")
    print(f"  Candy Reward: {spec.candy_reward}")
    print(f"  Step Penalty: {spec.step_penalty}")
    print(f"  Max Steps: {spec.max_steps}")
    print()
    
    # Basic path analysis (Manhattan distances)
    print("=" * 60)
    print("Path Analysis:")
    print("=" * 60)
    start_x, start_y = spec.start
    goal_x, goal_y = spec.goal
    manhattan_dist = abs(goal_x - start_x) + abs(goal_y - start_y)
    print(f"  Manhattan Distance (Start → Goal): {manhattan_dist} steps")
    if candies:
        for i, (candy_x, candy_y) in enumerate(candies):
            dist_to_candy = abs(candy_x - start_x) + abs(candy_y - start_y)
            dist_candy_to_goal = abs(goal_x - candy_x) + abs(goal_y - candy_y)
            print(f"  Distance (Start → Candy{i}): {dist_to_candy} steps")
            print(f"  Distance (Candy{i} → Goal): {dist_candy_to_goal} steps")
            print(f"  Total via Candy{i}: {dist_to_candy + dist_candy_to_goal} steps")
    print()
    print("=" * 60)


def visualize_gridworld_html(spec: GridSpec, output_path: str = "gridworld_env.html"):
    """
    Visualize a GridWorld spec as an HTML page.
    
    Args:
        spec: GridSpec configuration
        output_path: Path to save the HTML file
    """
    width = spec.width
    height = spec.height
    
    # Create grid
    grid = [["." for _ in range(width)] for _ in range(height)]
    
    # Mark walls
    for wx, wy in spec.walls:
        grid[wy][wx] = "#"
    
    # Mark start
    sx, sy = spec.start
    grid[sy][sx] = "S"
    
    # Mark goal
    gx, gy = spec.goal
    grid[gy][gx] = "G"
    
    # Mark candy cells
    candies = getattr(spec, "candies", ()) or (() if spec.candy is None else (spec.candy,))
    for cx, cy in candies:
        grid[cy][cx] = "C"
    
    # Mark risk cell
    if spec.risk is not None:
        rx, ry = spec.risk
        grid[ry][rx] = "R"
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GridWorld Environment</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .grid {{
            display: inline-block;
            border: 2px solid #333;
            margin: 20px auto;
            font-family: monospace;
        }}
        .grid-row {{
            display: flex;
        }}
        .grid-cell {{
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 18px;
            border: 1px solid #ddd;
        }}
        .cell-empty {{ background-color: #fff; }}
        .cell-wall {{ background-color: #333; color: white; }}
        .cell-start {{ background-color: #4CAF50; color: white; }}
        .cell-goal {{ background-color: #f44336; color: white; }}
        .cell-candy {{ background-color: #FFEB3B; color: #333; }}
        .cell-risk {{ background-color: #FF9800; color: white; }}
        .info {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .info h2 {{
            margin-top: 0;
            color: #555;
        }}
        .info-item {{
            margin: 5px 0;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 30px;
            height: 30px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GridWorld Environment ({width}×{height})</h1>
        
        <div style="text-align: center;">
            <div class="grid">
"""
    
    # Generate grid HTML
    for y in range(height):
        html += '                <div class="grid-row">\n'
        for x in range(width):
            cell = grid[y][x]
            if cell == "S":
                html += f'                    <div class="grid-cell cell-start">{cell}</div>\n'
            elif cell == "G":
                html += f'                    <div class="grid-cell cell-goal">{cell}</div>\n'
            elif cell == "C":
                html += f'                    <div class="grid-cell cell-candy">{cell}</div>\n'
            elif cell == "R":
                html += f'                    <div class="grid-cell cell-risk">{cell}</div>\n'
            elif cell == "#":
                html += f'                    <div class="grid-cell cell-wall">{cell}</div>\n'
            else:
                html += f'                    <div class="grid-cell cell-empty">{cell}</div>\n'
        html += '                </div>\n'
    
    html += """            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color cell-start"></div>
                <span>Start (S)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color cell-goal"></div>
                <span>Goal (G)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color cell-candy"></div>
                <span>Candy (C)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color cell-wall"></div>
                <span>Wall (#)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color cell-empty"></div>
                <span>Empty (.)</span>
            </div>
"""
    
    if spec.risk is not None:
        html += """            <div class="legend-item">
                <div class="legend-color cell-risk"></div>
                <span>Risk (R)</span>
            </div>
"""
    
    html += """        </div>
        
        <div class="info">
            <h2>Environment Configuration</h2>
            <div class="info-item"><strong>Start Position:</strong> """ + str(spec.start) + """</div>
            <div class="info-item"><strong>Goal Position:</strong> """ + str(spec.goal) + """</div>
"""
    
    if spec.candy is not None:
        html += f'            <div class="info-item"><strong>Candy Position:</strong> {spec.candy}</div>\n'
    if spec.risk is not None:
        html += f'            <div class="info-item"><strong>Risk Position:</strong> {spec.risk} (failure probability: {spec.risk_p})</div>\n'
    
    html += f"""            <div class="info-item"><strong>Goal Reward:</strong> {spec.goal_reward}</div>
            <div class="info-item"><strong>Candy Reward:</strong> {spec.candy_reward}</div>
            <div class="info-item"><strong>Step Penalty:</strong> {spec.step_penalty}</div>
            <div class="info-item"><strong>Max Steps:</strong> {spec.max_steps}</div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML visualization saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a GridWorld environment (loaded from env_specs/)."
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=1,
        help="Environment case_id (env_specs/case_{id}.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment random seed (only affects stochastic events like risk).",
    )
    args = parser.parse_args()

    env = make_env(seed=args.seed, case_id=args.case_id)
    spec = env.env.spec  # unwrap ObservationWrapper -> GridWorldEnv -> spec
    
    # Text visualization
    visualize_gridworld_text(spec)
    
    # HTML visualization
    visualize_gridworld_html(spec, output_path=f"gridworld_env_case_{args.case_id}.html")
    print("\n✓ GridWorld visualization complete!")
