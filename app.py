from dash import Dash, dcc, html  
import dash_bootstrap_components as dbc
import dash

# Initialize Dash app
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

nav_bar = html.Nav(
    children=[
        dcc.Link(
            "Overview",
            href="/",
            className="link",
            id="overview-link",
        ),
        dcc.Link(
            "Optimization",
            href="/optimization",
            className="link",
            style={"margin-left": "15px"},
            id="analysis-1-link",
        ),

        dcc.Link(
            "Model",
            href="/model",
            className="link",
            id="model-link",
            style={"margin-left": "20px"},

        ),
        dcc.Link(
            "About Us",
            href="/about_us",
            className="link",
            id="about-us-link",
            style={"margin-right": "0px"},
        ),
    ],
    style={
        "background-color": "white",
        "height": "70px",
        "display": "flex",
        "align-items": "center",
        "justify-content": "space-between",
        "padding": "0 5vw",  # Add padding to increase horizontal stretch
        "width": "55%",  # Set the width to 100% to stretch horizontally
        "max-width": "1200px",  # Set a maximum width to limit the stretching
        "margin": "0 auto",  # Center the nav bar horizontally
    },
)


app.layout = html.Div(
    [   
        html.Link(
            rel="stylesheet",
            href="/assets/styles.css"  # Path to the CSS file
        ),
        dcc.Location(id="url", refresh=False),
        nav_bar,
        dash.page_container
    ]
)


@app.callback(
    [dash.dependencies.Output(f"{link_id}", "className") for link_id in ["overview-link", "analysis-1-link", "analysis-2-link", "model-link", "about-us-link"]],
    [dash.dependencies.Input("url", "pathname")]
)
def update_active_link(pathname):
    return ["link active" if f"/{link_id.replace('-link', '')}" == pathname or link_id == "overview-link" and pathname == "/" else "link" for link_id in ["overview-link", "analysis-1-link", "analysis-2-link", "model-link", "about_us-link"]]

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=False)
    # If running locally instead of on Docker Container, access app at http://localhost:8080/