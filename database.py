import sqlite3


class Database:
    def __init__(self, path="conversations.db"):
        self.connection = sqlite3.connect(path)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                question_id INTEGER NOT NULL,
                version TEXT NOT NULL
            )
        """)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                agent TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id),

                UNIQUE (conversation_id, sequence)
            )
        """)

        self.connection.commit()

    def create_conversation(self, question_id, version):
        cursor = self.connection.execute(
            """
            INSERT INTO conversations (question_id, version)
            VALUES (?, ?)
            """,
            (question_id, version),
        )

        self.connection.commit()

        return cursor.lastrowid

    def add_message(self, conversation_id, sequence, agent, content):
        self.connection.execute(
            """
            INSERT INTO messages (
                conversation_id,
                sequence,
                agent,
                content
            )
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, sequence, agent, content),
        )

        self.connection.commit()

    def get_messages(self, conversation_id):
        return self.connection.execute(
            """
            SELECT sequence, agent, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sequence
            """,
            (conversation_id,),
        ).fetchall()
