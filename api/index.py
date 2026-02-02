from app import create_app

app = create_app()

# Export the app for Vercel
handler = app
