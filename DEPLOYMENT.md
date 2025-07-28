# Deploy Your Game App to Railway (FREE)

## 🚀 Quick Deploy to Railway

Railway is a free hosting platform perfect for Python Flask apps. Here's how to deploy your game app:

### Step 1: Push to GitHub
1. First, make sure all your files are committed to your GitHub repository
2. Your repository should contain:
   - `app.py` (main Flask application)
   - `requirements.txt` (Python dependencies)
   - `Procfile` (tells Railway how to run your app)
   - `webapp/` folder (contains your HTML, CSS, and JS files)

### Step 2: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up/login with your GitHub account
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will automatically detect it's a Python app and deploy it!

### Step 3: Access Your Game
- Once deployed, Railway will give you a URL like: `https://your-app-name.railway.app`
- Your games will be accessible at that URL
- The game search/selection page will load automatically

## 🎮 How It Works

### Game Features:
- **Game Search**: Search for games by score on the main page
- **4 Different Games**: AllyCraft, EcoSync, Elemental Synergy Quest, Tile Strategy Quest
- **Interactive Gameplay**: Use arrow keys or click buttons to play
- **Real-time Canvas**: Games render in real-time with colored tiles
- **Game Descriptions**: Each game has detailed mechanics and win conditions

### Game Controls:
- **Arrow Keys**: Move your character
- **WASD**: Control allies (in applicable games)
- **Action Buttons**: Special abilities (varies by game)
- **Reset Button**: Restart current game

## 📁 File Structure
```
your-project/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies  
├── Procfile           # Railway deployment config
├── webapp/            # Frontend files
│   ├── index.html     # Game player interface
│   ├── game_search.html # Game selection page
│   ├── styles.css     # All styling
│   └── js/
│       └── game.js    # Game rendering and interaction
└── DEPLOYMENT.md      # This file
```

## 🛠️ Development

To run locally:
```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser.

## 🎯 Game Details

### AllyCraft (Score: 0.5)
- Tactical RPG with ally management
- Control an archer and up to 3 allied archers
- Reach treasure chests to win

### EcoSync (Score: 0.09)  
- Environmental restoration adventure
- Cleanse pollution and manage ecosystems
- Multiple victory paths

### Elemental Synergy Quest (Score: 0.31)
- Grid-based cooperative adventure
- Elemental interactions and synergy mechanics
- Collect items to win

### Tile Strategy Quest (Score: 0.44)
- Tactical dungeon exploration
- Alchemy and crafting system
- Multiple victory conditions

## 💡 Tips
- Railway free tier gives you 512MB RAM and 1GB storage
- Your app will sleep after 30 minutes of inactivity but wake up when accessed
- Deployment is automatic - just push to GitHub and Railway updates!

Enjoy your deployed game app! 🎮 