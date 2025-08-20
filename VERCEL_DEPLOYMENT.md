# 🚀 Vercel Deployment Guide

## Overview

This guide will help you deploy your AI chatbot to Vercel with both frontend and backend components.

## 📋 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Push your code to GitHub
3. **API Keys**: Gather your AI service API keys

## 🔧 Deployment Options

### Option 1: Full-Stack Deployment (Recommended)

Deploy both frontend and backend to Vercel using serverless functions.

#### Step 1: Prepare Your Project

1. **Update requirements.txt for Vercel**:
   ```bash
   cp requirements-vercel.txt requirements.txt
   ```

2. **Set up environment variables**:
   Copy your API keys from `.env` - you'll need them for Vercel

#### Step 2: Deploy to Vercel

1. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "Import Project"
   - Connect your GitHub repository
   - Select your chatbot repository

2. **Configure Environment Variables**:
   In Vercel dashboard, go to Settings > Environment Variables and add:
   ```
   GOOGLE_AI_GENERATIVE=your_google_ai_api_key
   GROQ_API_KEY=your_groq_api_key
   GROQ_BASE_URL=https://api.groq.com/openai/v1
   DEEPSEEK_API_KEY=your_deepseek_api_key (optional)
   DEEPSEEK_BASE_URL=https://api.deepseek.com (optional)
   ```

3. **Deploy**:
   - Click "Deploy"
   - Vercel will automatically build and deploy your project

#### Step 3: Test Your Deployment

- Visit your Vercel URL (e.g., `your-project.vercel.app`)
- Test the chat interface with both Groq and Google AI
- Check the API endpoints at `your-project.vercel.app/health`

### Option 2: Frontend Only Deployment

Deploy just the frontend to Vercel and use a separate backend service.

#### Step 1: Prepare Frontend

1. **Use the simplified frontend**:
   ```bash
   # The public/index.html is already optimized for this
   ```

2. **Update API endpoints**:
   - Deploy your FastAPI backend to Railway, Render, or Heroku
   - Update the API_BASE URL in the frontend

#### Step 2: Deploy Frontend

1. **Create vercel.json for frontend only**:
   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "public/**",
         "use": "@vercel/static"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "/public/$1"
       }
     ]
   }
   ```

## 🛠️ Configuration Files

### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/main.py",
      "use": "@vercel/python"
    },
    {
      "src": "public/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/main.py"
    },
    {
      "src": "/(.*)",
      "dest": "/public/$1"
    }
  ]
}
```

### requirements.txt (for Vercel)
```
fastapi==0.116.0
uvicorn==0.35.0
python-dotenv==1.0.0
pydantic==2.11.7
httpx==0.25.2
aiofiles==23.2.1
```

## 🔐 Environment Variables Setup

### Required Variables:
- `GOOGLE_AI_GENERATIVE`: Your Google AI API key
- `GROQ_API_KEY`: Your Groq API key

### Optional Variables:
- `DEEPSEEK_API_KEY`: DeepSeek API key
- `MONGODB_URL`: MongoDB Atlas connection string
- `LANGFUSE_SECRET_KEY`: Langfuse monitoring key
- `LANGFUSE_PUBLIC_KEY`: Langfuse public key

## 🚨 Important Notes

### Limitations on Vercel:
1. **Serverless Functions**: Each function has a 10-second timeout limit
2. **File System**: No persistent file storage (use MongoDB Atlas for data)
3. **Memory**: Limited to 1GB RAM per function
4. **Cold Starts**: First request may be slower

### Optimizations:
1. **Lightweight Dependencies**: Use minimal required packages
2. **MongoDB Atlas**: Use cloud database instead of local MongoDB
3. **Error Handling**: Implement robust error handling for timeouts
4. **Caching**: Use Redis or Vercel's edge caching

## 📱 Testing Your Deployment

### Health Check:
```bash
curl https://your-project.vercel.app/health
```

### Chat API:
```bash
curl -X POST https://your-project.vercel.app/api/chat/groq \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "user_id": "test"}'
```

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**:
   - Check requirements.txt has all dependencies
   - Ensure Python path is correct in api/main.py

2. **Timeout Errors**:
   - Reduce model response length
   - Implement request timeouts

3. **Environment Variables**:
   - Verify all required environment variables are set in Vercel
   - Check variable names match exactly

### Debug Steps:
1. Check Vercel function logs
2. Test API endpoints individually
3. Verify environment variables
4. Check network connectivity

## 🎉 Success!

Once deployed, your chatbot will be available at:
- **Frontend**: `https://your-project.vercel.app`
- **API**: `https://your-project.vercel.app/api/`
- **Health Check**: `https://your-project.vercel.app/health`

Your AI chatbot is now live and accessible worldwide! 🌍✨
