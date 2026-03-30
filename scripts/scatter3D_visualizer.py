from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
import random
import zmq
import json
import signal
import threading
from flask import request





# Parameters
endpoint = "tcp://localhost:6000"
topic = "SKEL"
running = True
scene = dict(
        xaxis = dict(nticks=10, range=[-1,1],),
        yaxis = dict(nticks=10, range=[-1,1],),
        zaxis = dict(nticks=10, range=[-1,1],)
        )
camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.1),
        eye=dict(x=1.5, y=1.5, z=0.1)
        )

app = Dash(__name__)

data = None




def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with Werkzeug Server')
    func()

@app.server.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'



class SkeletonVisualizer:
    def __init__(self, socket):
        self.started = False
        self.socket = socket
        self.data = None
        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self.data_receiver, args=())

    def start(self):
        if self.started:
            return
        self.started = True
        self.thread.start()
        return self

    def data_receiver(self):
        global running, data
        while running:
            topic, message = self.socket.recv_string().split(" ", 1)
            array = json.loads(message)
            # print(f"Received: {array}")
            with self.mutex:
                self.data = array
                data = array

    def read_frame(self):
        with self.mutex:
            frame = self.data.copy() if self.data is not None else None
        return frame
    
    

    def load_interface(self):
        app.run(debug=True)




@app.callback(Output("graph", "figure"), Input('interval-component', 'n_intervals'))
def update_plot(n_intervals):   
    global running, data
    # with self.mutex:
    print([pnt[0]] for pnt in data if data is not None)


    x = [random.random(), random.random(), random.random()]
    y = [random.random(), random.random(), random.random()]
    z = [random.random(), random.random(), random.random()]
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])

    fig.update_layout(scene_aspectmode='cube', height=1200, width=1500, margin=dict(r=20, l=20, b=10, t=10))
    fig.update_layout(scene=scene, scene_camera=camera)

    return fig




# Main loop to receive data via ZeroMQ and update the plot
def main():
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.connect(endpoint)

    # Subscribe to "news" topic (prefix match)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    app.layout = html.Div([
    html.H4('Skeleton tracking 3D scatter'),
    dcc.Graph(id="graph"),
    dcc.Interval(
            id='interval-component',
            interval=100, # in milliseconds
            n_intervals=0)
    ], 
    id = "change-height", 
    style={'width': '100%', 'display': 'inline-block', 'height': '100%'})

    # Gestione segnali per chiusura pulita (es. CTRL+C o kill da script bash)
    def signal_handler(sig, frame):
        global running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Launch data receiver thread and load the Dash interface
    vis = SkeletonVisualizer(socket).start()
    #vis.load_interface()
    print("Sta partendo")
    app.run(debug=True, port=8001)
    print("Partito")


    
    

    # socket.close()



if __name__ == "__main__":
    main()
    