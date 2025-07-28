# Deploy Your Game App to Railway (FREE)

## 🚀 **Perfect Setup - No Interference with Your Main Website!**

Your Flask app is designed to **NOT touch your main website at all**. Here's what happens:

- **Your main website**: `yoursite.com/` - **completely unchanged**
- **Your games**: `yoursite.com/games/` - **new games section**

## 🌐 **Quick Deploy to Railway**

### Step 1: Push to GitHub
1. Make sure all files are committed to your GitHub repository
2. Your repository should contain:
   - `app.py` (Flask app with games at `/games/` path)
   - `requirements.txt` (simplified dependencies)
   - `Procfile` (Railway deployment config)
   - `webapp/` folder (game frontend files)
   - `game_app/cache/001/` (your game files and JSON)

### Step 2: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up/login with your GitHub account
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway automatically detects Python and deploys!

### Step 3: Access Your Games
Once deployed, you'll get a URL like: `https://your-app-name.railway.app`

**Your website structure:**
- **Main page**: `https://your-app-name.railway.app/` (your existing website)
- **Games hub**: `https://your-app-name.railway.app/games/`
- **Individual games**: `https://your-app-name.railway.app/games/load_game/0_AllyCraft_g1`

## 🎮 **How It Works**

### **No Conflicts with Main Site**
- Your `index.html`, `assets/`, `images/` etc. work exactly as before
- Games are completely separate at `/games/` path
- Your main website navigation and functionality unchanged

### **Dynamic Game Loading**
- **Add games**: Drop `.py` files in `game_app/cache/001/`
- **Update scores**: Edit `game_app/cache/001/game_scores.json`
- **Update descriptions**: Edit `game_app/cache/001/game_descriptions.json`
- **Deploy**: Push changes to GitHub - Railway auto-updates!

## 🔧 **Adding New Games** (Your Workflow)

### **Step 1: Add Game File**
```bash
# Add your new game to the cache folder
game_app/cache/001/MyNewGame_g5.py
```

### **Step 2: Update Scores**
Edit `game_app/cache/001/game_scores.json`:
```json
{
    "0_AllyCraft_g1": 0.5,
    "EcoSync_g2": 0.09,
    "Elemental_Synergy_Quest_g3": 0.31,
    "Tile_Strategy_Quest_g4": 0.44,
    "MyNewGame_g5": 0.75
}
```

### **Step 3: Update Descriptions**
Edit `game_app/cache/001/game_descriptions.json`:
```json
{
    "existing games...": "...",
    "MyNewGame_g5": {
        "Game narrative": "Your game description here...",
        "Win condition": "How to win your game..."
    }
}
```

### **Step 4: Deploy**
```bash
git add .
git commit -m "Added MyNewGame_g5"
git push
```
Railway automatically deploys the update!

## 🎯 **Game Access URLs**

After deployment, your games will be at:
- **Game selector**: `your-app.railway.app/games/`
- **AllyCraft**: `your-app.railway.app/games/load_game/0_AllyCraft_g1`
- **EcoSync**: `your-app.railway.app/games/load_game/EcoSync_g2`
- **Elemental Quest**: `your-app.railway.app/games/load_game/Elemental_Synergy_Quest_g3`
- **Tile Strategy**: `your-app.railway.app/games/load_game/Tile_Strategy_Quest_g4`

## 🛡️ **Robust Features**

- ✅ **Graceful Fallbacks**: Games work even if dependencies fail
- ✅ **Error Handling**: Won't crash your site
- ✅ **Dynamic Loading**: Automatically picks up new games
- ✅ **Simplified Dependencies**: No complex packages that cause deployment issues
- ✅ **Main Site Protection**: Your existing website completely untouched

## 📁 **File Structure**
```
your-project/
├── index.html          # Your main website (unchanged)
├── assets/             # Your existing assets (unchanged)  
├── images/             # Your existing images (unchanged)
├── app.py              # Flask app (serves games at /games/)
├── requirements.txt    # Minimal dependencies
├── Procfile           # Railway config
├── webapp/            # Game frontend
│   ├── index.html     # Game player
│   ├── game_search.html # Game selector
│   ├── styles.css     # Game styling
│   └── js/game.js     # Game engine
└── game_app/cache/001/
    ├── *.py           # Your game files
    ├── game_scores.json    # ← Edit to add games
    └── game_descriptions.json # ← Edit descriptions
```

## 💡 **Development**

Test locally:
```bash
python app.py
```

Then visit:
- `http://localhost:5000/` - Your main site
- `http://localhost:5000/games/` - Your games

## 🚀 **Ready to Deploy!**

The setup is **perfect for Railway deployment** and **won't interfere with your main website at all**. Your existing site works exactly as before, and games are accessible at the separate `/games/` path.

Deploy now and enjoy your games! 🎮 