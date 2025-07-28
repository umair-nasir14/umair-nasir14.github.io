class GameRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tileSize = 32;  // Set tile size to 32x32 for better visibility
        this.tileImages = {};
        this.gameState = null;
        this.mechanics = {};
        this.keyToAction = {};
        this.actionToKey = {};
    }

    async initialize() {
        try {
            // Get initial game state
            const response = await fetch('/get_game_state');
            const data = await response.json();
            console.log('Game state:', data);  // Debug log
            
            this.gameState = data;
            
            // Set canvas size based on map dimensions
            const mapHeight = this.gameState.map.length;
            const mapWidth = this.gameState.map[0].length;
            
            this.canvas.width = mapWidth * this.tileSize;
            this.canvas.height = mapHeight * this.tileSize;
            
            // Load tile images
            await this.loadTileImages(data.tile_paths);
            
            // Load game mechanics and setup controls
            await this.loadGameMechanics();
            
            // Load game description
            await this.loadGameDescription();
            
            // Initial render
            this.render();
        } catch (error) {
            console.error('Initialization error:', error);
        }
    }

    async loadGameMechanics() {
        try {
            const response = await fetch('/get_game_mechanics');
            const data = await response.json();
            
            if (data.mechanics) {
                this.mechanics = data.mechanics;
                this.createDynamicKeyMappings();
                this.createDynamicControls();
                this.displayControlsInfo();
                this.setupKeyboardControls();
            } else {
                console.error('Failed to load mechanics:', data.error);
                this.createFallbackControls();
            }
        } catch (error) {
            console.error('Error loading mechanics:', error);
            this.createFallbackControls();
        }
    }

    createDynamicKeyMappings() {
        // Define key preferences for different action types
        const keyPreferences = {
            // Player movement actions (Arrow keys)
            'move_player_up': 'ArrowUp',
            'move_player_down': 'ArrowDown', 
            'move_player_left': 'ArrowLeft',
            'move_player_right': 'ArrowRight',
            'move_player': 'ArrowUp', // Fallback for generic movement
            
            // Ally movement actions (WASD keys)
            'move_ally_up': 'KeyW',
            'move_ally_down': 'KeyS',
            'move_ally_left': 'KeyA',
            'move_ally_right': 'KeyD',
            
            // Common action keys
            'Space': 'Space',
            'KeyE': 'KeyE',
            'KeyR': 'KeyR',
            'KeyF': 'KeyF',
            'KeyQ': 'KeyQ',
            'KeyT': 'KeyT',
            'KeyG': 'KeyG',
            'KeyH': 'KeyH',
            'KeyZ': 'KeyZ',
            'KeyX': 'KeyX',
            'KeyC': 'KeyC',
            'KeyV': 'KeyV',
            'KeyB': 'KeyB',
            'KeyN': 'KeyN'
        };

        // Available keys for assignment (excluding WASD which are now reserved for ally movement)
        const availableKeys = [
            'Space', 'KeyE', 'KeyR', 'KeyF', 'KeyQ', 'KeyT', 'KeyG', 
            'KeyH', 'KeyZ', 'KeyX', 'KeyC', 'KeyV', 'KeyB', 'KeyN'
        ];
        
        let keyIndex = 0;
        this.keyToAction = {};
        this.actionToKey = {};

        // First, assign all movement keys (both player and ally)
        const playerMovements = ['move_player_up', 'move_player_down', 'move_player_left', 'move_player_right'];
        const allyMovements = ['move_ally_up', 'move_ally_down', 'move_ally_left', 'move_ally_right'];
        const allMovements = [...playerMovements, ...allyMovements];
        
        allMovements.forEach(mechanic => {
            if (this.mechanics[mechanic] !== undefined) {
                const action = this.mechanics[mechanic];
                const key = keyPreferences[mechanic];
                if (key) {
                    this.keyToAction[key] = action;
                    this.actionToKey[action] = key;
                }
            }
        });

        // Then assign other actions
        Object.entries(this.mechanics).forEach(([mechanic, action]) => {
            // Skip if already assigned (like movement keys)
            if (this.actionToKey[action] !== undefined) return;
            
            // Try to use preferred key if available
            let preferredKey = keyPreferences[mechanic];
            if (preferredKey && !this.keyToAction[preferredKey]) {
                this.keyToAction[preferredKey] = action;
                this.actionToKey[action] = preferredKey;
            } else {
                // Assign next available key
                while (keyIndex < availableKeys.length && this.keyToAction[availableKeys[keyIndex]]) {
                    keyIndex++;
                }
                if (keyIndex < availableKeys.length) {
                    const key = availableKeys[keyIndex];
                    this.keyToAction[key] = action;
                    this.actionToKey[action] = key;
                    keyIndex++;
                }
            }
        });

        console.log('Key mappings:', this.keyToAction);
        console.log('Action mappings:', this.actionToKey);
    }

    createDynamicControls() {
        const actionControls = document.getElementById('actionControls');
        actionControls.innerHTML = '';

        // Collect all movement actions and display them as arrow keys
        const allMovementKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'KeyW', 'KeyA', 'KeyS', 'KeyD'];
        const movementActions = {};
        
        // Map movement keys to their directional actions
        allMovementKeys.forEach(key => {
            if (this.keyToAction[key] !== undefined) {
                const direction = this.getMovementDirection(key);
                if (!movementActions[direction]) {
                    movementActions[direction] = [];
                }
                movementActions[direction].push({
                    key: key,
                    action: this.keyToAction[key]
                });
            }
        });

        // Create arrow key controls for all movement actions
        if (Object.keys(movementActions).length > 0) {
            const movementContainer = document.createElement('div');
            movementContainer.className = 'arrow-controls';
            
            // Create up arrows
            if (movementActions['up']) {
                const upContainer = document.createElement('div');
                upContainer.className = 'direction-group';
                
                movementActions['up'].forEach((movement, index) => {
                    const upBtn = document.createElement('button');
                    upBtn.textContent = '↑';
                    upBtn.onclick = () => this.takeAction(movement.action);
                    upBtn.title = `Move Up (${this.getKeyDisplayName(movement.key)})`;
                    if (index > 0) upBtn.style.marginLeft = '5px';
                    upContainer.appendChild(upBtn);
                });
                
                movementContainer.appendChild(upContainer);
            }

            // Horizontal controls container
            const horizontalDiv = document.createElement('div');
            horizontalDiv.className = 'horizontal-controls';
            
            // Create left arrows
            if (movementActions['left']) {
                const leftContainer = document.createElement('div');
                leftContainer.className = 'direction-group';
                leftContainer.style.display = 'inline-flex';
                leftContainer.style.flexDirection = 'column';
                leftContainer.style.gap = '5px';
                
                movementActions['left'].forEach(movement => {
                    const leftBtn = document.createElement('button');
                    leftBtn.textContent = '←';
                    leftBtn.onclick = () => this.takeAction(movement.action);
                    leftBtn.title = `Move Left (${this.getKeyDisplayName(movement.key)})`;
                    leftContainer.appendChild(leftBtn);
                });
                
                horizontalDiv.appendChild(leftContainer);
            }

            // Create down arrows
            if (movementActions['down']) {
                const downContainer = document.createElement('div');
                downContainer.className = 'direction-group';
                downContainer.style.display = 'inline-flex';
                downContainer.style.flexDirection = 'column';
                downContainer.style.gap = '5px';
                
                movementActions['down'].forEach(movement => {
                    const downBtn = document.createElement('button');
                    downBtn.textContent = '↓';
                    downBtn.onclick = () => this.takeAction(movement.action);
                    downBtn.title = `Move Down (${this.getKeyDisplayName(movement.key)})`;
                    downContainer.appendChild(downBtn);
                });
                
                horizontalDiv.appendChild(downContainer);
            }

            // Create right arrows
            if (movementActions['right']) {
                const rightContainer = document.createElement('div');
                rightContainer.className = 'direction-group';
                rightContainer.style.display = 'inline-flex';
                rightContainer.style.flexDirection = 'column';
                rightContainer.style.gap = '5px';
                
                movementActions['right'].forEach(movement => {
                    const rightBtn = document.createElement('button');
                    rightBtn.textContent = '→';
                    rightBtn.onclick = () => this.takeAction(movement.action);
                    rightBtn.title = `Move Right (${this.getKeyDisplayName(movement.key)})`;
                    rightContainer.appendChild(rightBtn);
                });
                
                horizontalDiv.appendChild(rightContainer);
            }
            
            movementContainer.appendChild(horizontalDiv);
            actionControls.appendChild(movementContainer);
        }

        // Create non-movement action controls
        const nonMovementActions = Object.entries(this.keyToAction).filter(([key, action]) => 
            !allMovementKeys.includes(key)
        );

        if (nonMovementActions.length > 0) {
            const otherActionsDiv = document.createElement('div');
            otherActionsDiv.className = 'other-actions';

            nonMovementActions.forEach(([key, action]) => {
                const btn = document.createElement('button');
                btn.textContent = this.getKeyDisplayName(key);
                btn.onclick = () => this.takeAction(action);
                btn.classList.add('action-btn');
                otherActionsDiv.appendChild(btn);
            });

            actionControls.appendChild(otherActionsDiv);
        }
    }

    getMovementDirection(key) {
        const directionMap = {
            'ArrowUp': 'up',
            'KeyW': 'up',
            'ArrowDown': 'down', 
            'KeyS': 'down',
            'ArrowLeft': 'left',
            'KeyA': 'left',
            'ArrowRight': 'right',
            'KeyD': 'right'
        };
        return directionMap[key];
    }

    getKeyDisplayName(key) {
        const displayNames = {
            'Space': 'Space',
            'KeyW': 'W',
            'KeyA': 'A',
            'KeyS': 'S',
            'KeyD': 'D',
            'KeyE': 'E',
            'KeyR': 'R', 
            'KeyF': 'F',
            'KeyQ': 'Q',
            'KeyT': 'T',
            'KeyG': 'G',
            'KeyH': 'H',
            'KeyZ': 'Z',
            'KeyX': 'X',
            'KeyC': 'C',
            'KeyV': 'V',
            'KeyB': 'B',
            'KeyN': 'N'
        };
        return displayNames[key] || key;
    }

    displayControlsInfo() {
        const controlsInfo = document.getElementById('controlsInfo');
        controlsInfo.innerHTML = '';

        Object.entries(this.mechanics).forEach(([mechanic, action]) => {
            const key = this.actionToKey[action];
            if (key) {
                const controlItem = document.createElement('div');
                controlItem.className = 'control-item';
                
                const keySpan = document.createElement('span');
                keySpan.className = 'control-key';
                keySpan.textContent = this.getKeyDisplayName(key);
                
                const descSpan = document.createElement('span');
                descSpan.className = 'control-description';
                descSpan.textContent = this.formatMechanicName(mechanic);
                
                controlItem.appendChild(keySpan);
                controlItem.appendChild(descSpan);
                controlsInfo.appendChild(controlItem);
            }
        });
    }

    formatMechanicName(mechanic) {
        // Convert snake_case to readable format
        return mechanic
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    async loadGameDescription() {
        try {
            const response = await fetch('/get_game_description');
            const data = await response.json();
            
            if (data.narrative || data.win_condition) {
                this.displayGameDescription(data);
            } else {
                console.error('Failed to load game description:', data.error);
                this.displayFallbackDescription();
            }
        } catch (error) {
            console.error('Error loading game description:', error);
            this.displayFallbackDescription();
        }
    }

    displayGameDescription(description) {
        const gameDescriptionEl = document.getElementById('gameDescription');
        
        let html = '';
        
        if (description.narrative) {
            html += `
                <div class="narrative">
                    <h3>📖 Game Story</h3>
                    <p>${description.narrative}</p>
                </div>
            `;
        }
        
        if (description.win_condition) {
            html += `
                <div class="win-condition">
                    <h3>🎯 How to Win</h3>
                    <p>${description.win_condition}</p>
                </div>
            `;
        }
        
        gameDescriptionEl.innerHTML = html;
    }

    displayFallbackDescription() {
        const gameDescriptionEl = document.getElementById('gameDescription');
        gameDescriptionEl.innerHTML = '<p>Game description not available.</p>';
    }

    createFallbackControls() {
        // Fallback to basic movement controls if mechanics loading fails
        const actionControls = document.getElementById('actionControls');
        actionControls.innerHTML = `
            <div class="arrow-controls">
                <button onclick="window.game.takeAction(0)">↑</button>
                <div class="horizontal-controls">
                    <button onclick="window.game.takeAction(2)">←</button>
                    <button onclick="window.game.takeAction(1)">↓</button>
                    <button onclick="window.game.takeAction(3)">→</button>
                </div>
            </div>
            <div class="other-actions" style="margin-top: 15px; display: flex; gap: 10px; justify-content: center;">
                <button onclick="window.game.takeAction(4)">Space</button>
            </div>
        `;

        const controlsInfo = document.getElementById('controlsInfo');
        controlsInfo.innerHTML = '<p>Using fallback controls. Check console for errors.</p>';
    }

    async loadTileImages(tilePaths) {
        console.log('Loading tile images:', tilePaths);  // Debug log
        for (const [char, path] of Object.entries(tilePaths)) {
            try {
                const img = new Image();
                img.src = path;
                await new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = () => {
                        console.warn(`Failed to load image for ${char} from ${path}, using placeholder`);
                        resolve(); // Don't reject, just use placeholder
                    };
                });
                this.tileImages[char] = img;
                console.log(`Loaded image for ${char}`);  // Debug log
            } catch (error) {
                console.error(`Error loading tile ${char}:`, error);
            }
        }
    }

    render() {
        if (!this.gameState || !this.gameState.map) {
            console.error('No game state to render');
            return;
        }
    
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const map = this.gameState.map;
        
        // Render the game map with colored tiles
        for (let y = 0; y < map.length; y++) {
            for (let x = 0; x < map[y].length; x++) {
                const tile = map[y][x];
                
                // Set color based on tile type
                this.ctx.fillStyle = this.getTileColor(tile);
                this.ctx.fillRect(x * this.tileSize, y * this.tileSize, this.tileSize, this.tileSize);
                
                // Add border
                this.ctx.strokeStyle = '#333';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(x * this.tileSize, y * this.tileSize, this.tileSize, this.tileSize);
                
                // Draw tile character
                this.ctx.fillStyle = '#fff';
                this.ctx.font = '16px monospace';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(
                    tile,
                    x * this.tileSize + this.tileSize / 2,
                    y * this.tileSize + this.tileSize / 2
                );
            }
        }
    }

    getTileColor(tile) {
        const colorMap = {
            'A': '#ecf0f1',  // Ground - light gray
            '@': '#3498db',  // Player - blue
            '#': '#e74c3c',  // Enemy - red
            '&': '#f39c12',  // NPC - orange
            'S': '#2ecc71',  // Ally/Shield - green
            'O': '#f39c12',  // Treasure - orange
            'R': '#9b59b6',  // Resource - purple
            'I': '#f1c40f',  // Item - yellow
            'M': '#95a5a6',  // Polluted - gray
            'G': '#27ae60',  // Clean - dark green
            'F': '#e67e22',  // Fire - orange-red
            'W': '#3498db',  // Water - blue
            'E': '#c0392b',  // Explosive - dark red
            'C': '#8e44ad',  // Clone/Combination - purple
            'T': '#1abc9c',  // Teleport - teal
            'D': '#34495e',  // Mysterious - dark blue-gray
            'X': '#e8daef',  // Craft X - light purple
            'Y': '#e8daef',  // Craft Y - light purple
            'Z': '#e8daef',  // Craft Z - light purple
        };
        return colorMap[tile] || '#bdc3c7'; // Default gray
    }

    async takeAction(action) {
        try {
            const response = await fetch(`/take_action/${action}`);
            const data = await response.json();
            if (data.observation) {
                this.gameState.map = data.observation;
                this.render();
            }
            // Update score display
            if (data.reward !== undefined) {
                const currentScore = parseInt(document.getElementById('scoreValue').textContent) || 0;
                const newScore = currentScore + data.reward;
                document.getElementById('scoreValue').textContent = newScore;
            }
            return data;
        } catch (error) {
            console.error('Action error:', error);
        }
    }

    async reset() {
        try {
            const response = await fetch('/reset_game');
            const data = await response.json();
            if (data.initial_state) {
                this.gameState.map = data.initial_state;
                this.render();
                // Reset score display
                document.getElementById('scoreValue').textContent = '0';
            }
        } catch (error) {
            console.error('Reset error:', error);
        }
    }

    setupKeyboardControls() {
        // Add keyboard event listener
        document.addEventListener('keydown', (event) => {
            // Prevent default behavior for game controls
            if (this.keyToAction[event.code] !== undefined) {
                event.preventDefault();
                const action = this.keyToAction[event.code];
                this.takeAction(action);
                
                // Visual feedback - highlight corresponding button
                this.highlightButton(event.code);
            }
        });
    }

    highlightButton(keyCode) {
        // Find and highlight the button for visual feedback
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            const displayName = this.getKeyDisplayName(keyCode);
            const arrowSymbols = {'ArrowUp': '↑', 'ArrowDown': '↓', 'ArrowLeft': '←', 'ArrowRight': '→', 'KeyW': '↑', 'KeyS': '↓', 'KeyA': '←', 'KeyD': '→'};
            
            if (button.textContent === displayName || 
                (arrowSymbols[keyCode] && button.textContent === arrowSymbols[keyCode] && 
                 button.title && button.title.includes(this.getKeyDisplayName(keyCode)))) {
                button.classList.add('active');
                setTimeout(() => button.classList.remove('active'), 150);
            }
        });
    }
}

// Initialize when the page loads
let game; // Make game globally accessible
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('gameCanvas');
    if (!canvas) {
        console.error('Canvas element not found!');
        return;
    }

    game = new GameRenderer(canvas);
    window.game = game; // Make it globally accessible
    game.initialize();
}); 