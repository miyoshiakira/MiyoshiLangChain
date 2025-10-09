import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  TextField,
  Button,
  Box,
  Typography,
  CircularProgress,
  Paper,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import { createTheme, ThemeProvider, type Shadows } from '@mui/material/styles';

// --- Configuration ---
// The target API endpoint for the POST request
const API_ENDPOINT = 'https://rag-api-handler-xxv2n73tia-uc.a.run.app';

// MUI Theme setup for consistent design
const theme = createTheme({
  palette: {
    primary: {
      main: '#4F46E5', // Indigo-600
    },
    secondary: {
      main: '#10B981', // Emerald-500
    },
    background: {
      default: '#f3f4f6', // ソフトな背景色 (Tailwind gray-100)
    }
  },
  typography: {
    fontFamily: ['Inter', 'sans-serif'].join(','),
  },
  // BoxShadowをカスタムし、よりモダンな影を表現
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.1)',
    '0px 8px 16px rgba(0, 0, 0, 0.15)',
    '0px 10px 20px rgba(0, 0, 0, 0.18)',
    '0px 12px 24px rgba(0, 0, 0, 0.2)',
    '0px 14px 28px rgba(0, 0, 0, 0.22)',
    '0px 16px 32px rgba(0, 0, 0, 0.25)',
    '0px 20px 40px rgba(0, 0, 0, 0.3)',
    '0px 24px 48px rgba(0, 0, 0, 0.35)',
    '0px 30px 60px rgba(0, 0, 0, 0.4)',
    // ...
  ] as unknown as Shadows,
});


/**
 * APIレスポンスの構造を定義するインターフェースです。
 */
interface ApiResponse {
  // APIからの主要なテキスト応答
  answer: string;
  // 構造化されたデータなど、ここでは使用しませんが型定義は保持
  response: object;
}

/**
 * チャット履歴のメッセージの構造を定義します。
 */
interface Message {
  sender: 'user' | 'ai';
  text: string;
}

// Helper function to handle exponential backoff for API calls
// この関数は元のコードから変更していません。
const exponentialBackoffFetch = async (url: string, options: RequestInit, maxRetries: number = 3) => {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      if (response.status !== 429 && response.ok) {
        return response;
      }
      if (response.status === 429 && attempt < maxRetries - 1) {
        const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
        console.log(`Rate limit hit. Retrying in ${delay.toFixed(0)}ms (Attempt ${attempt + 1})...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }
      return response; // Return the response for non-429 errors or after final attempt
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;
      const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
      console.log(`Network error. Retrying in ${delay.toFixed(0)}ms (Attempt ${attempt + 1})...`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw new Error("API call failed after multiple retries.");
};

// --- Chat Bubble Component ---

/**
 * 個々のチャットメッセージを表示するコンポーネント
 */
const ChatBubble = ({ message }: { message: Message }) => {
  const isUser = message.sender === 'user';
  
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
        // アニメーションを追加
        opacity: 0,
        animation: 'fadeIn 0.5s ease-out forwards',
        '@keyframes fadeIn': {
          '0%': { opacity: 0, transform: 'translateY(10px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
      }}
    >
      <Paper
        elevation={2}
        sx={{
          p: { xs: 1.5, sm: 2 },
          maxWidth: { xs: '90%', sm: '75%' },
          borderRadius: isUser 
            ? '1.2rem 1.2rem 0.2rem 1.2rem' // ユーザー側は右下だけ角丸を減らす
            : '1.2rem 1.2rem 1.2rem 0.2rem', // AI側は左下だけ角丸を減らす
          backgroundColor: isUser 
            ? theme.palette.primary.main // ユーザーはメインカラー
            : theme.palette.grey[50], // AIは淡い色
          color: isUser ? 'white' : theme.palette.grey[900],
          boxShadow: isUser ? theme.shadows[3] : theme.shadows[1],
          transition: 'all 0.3s',
        }}
      >
        <Typography 
          component="div"
          sx={{ 
            whiteSpace: 'pre-wrap', // 改行を保持
            fontSize: { xs: '0.9rem', sm: '1rem' },
            lineHeight: 1.5,
          }}
        >
          {/* テキストコンテンツ */}
          {message.text}
        </Typography>
      </Paper>
    </Box>
  );
};


// --- Main Application Component ---

const AIChatApp = () => {
  // ユーザーの入力はテキストとして保持
  const [promptInput, setPromptInput] = useState<string>('');
  // チャット履歴を管理
  const [chatHistory, setChatHistory] = useState<Message[]>([
    { sender: 'ai', text: 'こんにちは！何について質問したいですか？RAG APIを利用して最新の情報を提供できます。' }
  ]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // チャット履歴の末尾に自動でスクロールするためのRef
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 履歴が更新されるたびにスクロール
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleSubmit = useCallback(async (event: React.FormEvent) => {
    event.preventDefault();
    
    const prompt = promptInput.trim();
    if (!prompt || isLoading) return;

    // 1. ユーザーメッセージを履歴に追加
    setChatHistory(prev => [...prev, { sender: 'user', text: prompt }]);
    
    // 2. 入力フィールドをクリア
    setPromptInput('');
    
    // 3. APIリクエストを開始
    setIsLoading(true);
    
    // APIへ送信するJSONボディを構築
    const requestBody = {
      prompt: prompt, 
    };

    try {
      // ** 既存のAPIコールロジックを保持 **
      const response = await exponentialBackoffFetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`APIリクエストが失敗しました。ステータスコード: ${response.status}. レスポンス: ${errorText.substring(0, 100)}...`);
      }

      const data: ApiResponse = await response.json(); 
      
      // 4. AIの応答を履歴に追加
      setChatHistory(prev => {
        return [...prev, { sender: 'ai', text: data.answer || 'APIからの応答が空でした。' }];
      });

    } catch (e) {
      console.error('Fetch error:', e);
      const errorMessage = `リクエスト処理中にエラーが発生しました: ${e instanceof Error ? e.message : String(e)}`;
      // エラーメッセージをチャット履歴にも追加して表示
      setChatHistory(prev => [...prev, { sender: 'ai', text: `[エラー] ${errorMessage}` }]);
    } finally {
      setIsLoading(false);
    }
  }, [promptInput, isLoading]);
  
  // Enterキーでの送信をハンドル (Shift + Enter で改行)
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event as unknown as React.FormEvent);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      {/* 全体コンテナ: 画面いっぱいに広げ、チャットUIを中央に配置 */}
      <Box 
        sx={{ 
          backgroundColor: theme.palette.background.default, 
          minHeight: '100vh', 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          p: { xs: 2, sm: 4 }
        }}
      >
        {/* チャットコンテナ (Paper) */}
        <Paper 
          elevation={8} 
          sx={{ 
            width: '100%',
            maxWidth: '800px', // 最大幅を設定
            height: '90vh', // 画面の高さの90%
            borderRadius: '1.5rem',
            boxShadow: theme.shadows[8],
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden', // 角丸を有効にするため
          }}
        >
          {/* ヘッダー */}
          <Box sx={{ 
            p: 3, 
            backgroundColor: theme.palette.primary.main, 
            color: 'white', 
            textAlign: 'center',
            borderTopLeftRadius: '1.5rem',
            borderTopRightRadius: '1.5rem',
            boxShadow: theme.shadows[4]
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <ChatBubbleOutlineIcon sx={{ mr: 1, fontSize: 32 }} />
                <Typography 
                    variant="h5" 
                    component="h1" 
                    sx={{ fontWeight: 700 }}
                >
                    AIナレッジチャット
                </Typography>
            </Box>
            <Typography variant="body2" sx={{ mt: 0.5, opacity: 0.8 }}>
                外部RAG APIと連携した対話インターフェース
            </Typography>
          </Box>

          {/* チャット履歴表示エリア */}
          <Box 
            sx={{ 
              flexGrow: 1, 
              overflowY: 'auto', 
              p: { xs: 2, sm: 4 },
              backgroundColor: 'white', 
            }}
          >
            {chatHistory.map((message, index) => (
              <ChatBubble key={index} message={message} />
            ))}
            
            {/* ローディングインジケーター */}
            {isLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                <Paper 
                  elevation={1} 
                  sx={{ p: 1.5, maxWidth: '80%', borderRadius: '1.2rem 1.2rem 1.2rem 0.2rem', backgroundColor: theme.palette.grey[100] }}
                >
                  <CircularProgress size={20} sx={{ color: theme.palette.primary.main, mr: 1 }} />
                  <Typography variant="body2" component="span" sx={{ color: theme.palette.grey[600] }}>AIが応答を作成中...</Typography>
                </Paper>
              </Box>
            )}

            {/* スクロール位置の基準点 */}
            <div ref={messagesEndRef} />
          </Box>
          
          {/* 入力フォームエリア */}
          <Box 
            component="form" 
            onSubmit={handleSubmit} 
            sx={{ 
              p: { xs: 2, sm: 3 }, 
              borderTop: `1px solid ${theme.palette.grey[200]}`,
              backgroundColor: theme.palette.grey[50], 
            }}
          >
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                label="質問を入力してください"
                multiline
                maxRows={4} 
                fullWidth
                value={promptInput}
                onChange={(e) => setPromptInput(e.target.value)}
                onKeyDown={handleKeyDown}
                variant="outlined"
                disabled={isLoading}
                placeholder='例: 最新の市場動向について教えてください'
                sx={{ 
                  backgroundColor: 'white', 
                  borderRadius: '0.75rem',
                  '.MuiOutlinedInput-root': {
                    paddingRight: '0 !important',
                    borderRadius: '0.75rem',
                  }
                }}
              />
              
              <Button
                type="submit"
                variant="contained"
                color="primary"
                size="large"
                disabled={isLoading || !promptInput.trim()}
                sx={{
                  minWidth: { xs: '140px', sm: '140px' },
                  px: { xs: 2, sm: 3 },
                  borderRadius: '0.75rem',
                  fontWeight: 'bold',
                  boxShadow: theme.shadows[4],
                  transition: 'all 0.3s',
                  '&:hover': {
                    boxShadow: theme.shadows[6],
                  },
                }}
              >
                {/* 画面サイズに応じてアイコンのみ/アイコン+テキストを切り替える */}
                <SendIcon sx={{ fontSize: { xs: 20, sm: 24 } }} />
                <Typography sx={{ display: { xs: 'none', sm: 'block' }, ml: 1 }}>送信</Typography>
              </Button>
            </Box>
          </Box>
        </Paper>
      </Box>
    </ThemeProvider>
  );
};

export default AIChatApp;
