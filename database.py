import json
import sqlite3


class DatabaseManager:
    def __init__(self, users_db_path='data/users.db', images_db_path='data/images.db'):
        self.conn_users = sqlite3.connect(users_db_path)
        self.conn_images = sqlite3.connect(images_db_path)
        self.c_users = self.conn_users.cursor()
        self.c_images = self.conn_images.cursor()

        self.create_tables()

    def create_tables(self):
        self.c_users.execute('''CREATE TABLE IF NOT EXISTS users
                                (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
        self.conn_users.commit()

        # self.c_images.execute('''CREATE TABLE IF NOT EXISTS images
        #                          (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, prompt TEXT, model TEXT, image BLOB)''')
        self.c_images.execute('''CREATE TABLE IF NOT EXISTS images
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, model_name TEXT, model_kwargs TEXT, image BLOB)''')
        self.conn_images.commit()

    def authenticate(self, username, password):
        self.c_users.execute(
            "SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = self.c_users.fetchone()
        if user:
            return True, user[0], user[1]
        else:
            return False, None, None

    def create_account(self, username, password):
        try:
            self.c_users.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            self.conn_users.commit()
        except:
            return "User name taken"

    def insert_image(self, user_id, model_name, model_kwargs, image):
        model_kwargs_json = json.dumps(model_kwargs)
        self.c_images.execute("INSERT INTO images (user_id, model_name, model_kwargs, image) VALUES (?, ?, ?, ?)",
                              (user_id, model_name, model_kwargs_json, image))
        self.conn_images.commit()

    def get_user_ids(self):
        self.c_users.execute("SELECT id FROM users")
        return self.c_users.fetchall()

    def get_users(self):
        self.c_users.execute("SELECT * FROM users")
        return self.c_users.fetchall()

    def get_user_images(self, user_id):
        self.c_images.execute("SELECT * FROM images WHERE user_id=?", (user_id,))
        return self.c_images.fetchall()

    def delete_user(self, user_id):
        self.c_users.execute("DELETE FROM users WHERE id=?", (user_id,))
        self.conn_users.commit()
        self.c_images.execute("DELETE FROM images WHERE user_id=?", (user_id,))
        self.conn_images.commit()

    def delete_images(self, image_ids):
        for image_id in image_ids:
            self.c_images.execute("DELETE FROM images WHERE id=?", (image_id,))
        self.conn_images.commit()

    def apocalypse(self):
        self.c_users.execute("DELETE FROM users")
        self.conn_users.commit()
        self.c_images.execute("DELETE FROM images")
        self.conn_images.commit()

if __name__ == "__main__":
    db_manager = DatabaseManager()
    db_manager.apocalypse()
    print("DONE!")
