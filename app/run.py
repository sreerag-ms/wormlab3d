if __name__ == '__main__':
    from app import app
    from wormlab3d import APP_PORT
    app.run(host='127.0.0.1', port=APP_PORT, debug=True)
