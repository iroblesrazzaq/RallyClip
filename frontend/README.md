# DeepMatch Frontend

A modern, responsive web interface for the DeepMatch AI video analysis system.

## Features

### 🎯 Core Functionality
- **Drag & Drop File Upload**: Intuitive file selection with visual feedback
- **Real-time Progress Tracking**: Live progress bars for each pipeline step
- **Modern UI/UX**: Beautiful, responsive design with smooth animations
- **Download Management**: Easy access to analysis results

### 📊 Pipeline Steps Tracked
1. **Pose Extraction** - Extract player poses from video frames
2. **Preprocessing** - Clean and prepare pose data
3. **Feature Engineering** - Generate analysis features
4. **AI Inference** - Run machine learning predictions
5. **Output Video Drawing** - Generate annotated output video

### 🎨 Design Features
- Responsive design (mobile-friendly)
- Drag and drop with visual feedback
- Real-time progress animations
- Modern gradient styling
- Accessible UI components
- Error handling and notifications

## Files Structure

```
frontend/
├── index.html          # Main HTML structure
├── styles.css          # Complete CSS styling
├── script.js           # JavaScript functionality
├── server.py           # Development server
└── README.md           # This file
```

## Quick Start

### Option 1: Development Server
```bash
cd frontend
python3 server.py
```
This will start a local server at `http://localhost:8080` and automatically open your browser.

### Option 2: Direct File Access
Simply open `index.html` in your web browser.

## API Integration

The frontend expects the following API endpoints (to be implemented in the backend):

### Upload and Start Analysis
```
POST /api/upload-and-start
Content-Type: multipart/form-data
Body: video file

Response: { "job_id": "unique-job-id" }
```

### Progress Monitoring
```
GET /api/progress/{job_id}

Response: {
  "status": "in_progress|completed|failed",
  "steps": {
    "pose": { "status": "completed", "progress": 100 },
    "preprocess": { "status": "in_progress", "progress": 45 },
    "feature": { "status": "waiting", "progress": 0 },
    "inference": { "status": "waiting", "progress": 0 },
    "output": { "status": "waiting", "progress": 0 }
  },
  "error": "error message if failed"
}
```

### Cancel Analysis
```
POST /api/cancel/{job_id}

Response: { "status": "cancelled" }
```

### Download Results
```
GET /api/download/video/{job_id}    # Returns MP4 file
GET /api/download/csv/{job_id}      # Returns CSV file
```

## Browser Compatibility

- ✅ Chrome 60+
- ✅ Firefox 60+
- ✅ Safari 12+
- ✅ Edge 79+

### Development Notes

### Key JavaScript Classes
- `DeepMatch`: Main application class handling all functionality
- File handling with drag/drop support
- Progress monitoring with automatic updates
- Download management with blob handling

### CSS Features
- CSS Grid and Flexbox layouts
- Custom animations and transitions
- Responsive breakpoints
- Modern color schemes and gradients

### Future Enhancements
- File validation improvements
- Batch processing support
- Analysis history
- Settings panel
- Video preview functionality

## Testing

The frontend includes comprehensive error handling and user feedback:

- File type validation
- File size limits (2GB)
- Network error handling
- Progress update failures
- Download error management

## Contributing

When extending the frontend:

1. Maintain responsive design principles
2. Add proper error handling
3. Include loading states for user feedback
4. Follow the existing CSS naming conventions
5. Test across different browsers and devices
