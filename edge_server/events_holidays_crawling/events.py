import mysql.connector

# Database connection
conn = mysql.connector.connect(
    host="localhost",  # or use "mysql-fyp" if running inside a Docker container
    user="root",
    password="rootpassword",
    database="mydb"
)

def extract_events():
    cursor = conn.cursor(dictionary=True)

    # Insert data
    query = """
    SELECT * FROM events;
    """

    cursor.execute(query)
    results = cursor.fetchall()  # Fetch all records
    
    cursor.close()  # Close cursor after execution
    return results  # Returns a list of dictionaries

'''
data = extract_events()
print(data)
'''