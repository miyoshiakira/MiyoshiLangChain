import React, { useState, useCallback, useMemo } from 'react';
import {
  TextField,
  Button,
  Box,
  Typography,
  Container,
  Alert,
  CircularProgress,
  Paper,
  Divider,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { createTheme, ThemeProvider } from '@mui/material/styles';

// --- Configuration ---
// The target API endpoint for the POST request
const API_ENDPOINT = 'https://rag-api-handler-xxv2n73tia-uc.a.run.app';

// Initial query JSON for demonstration
const initialQuery = JSON.stringify({
  prompt: "ジェミニを使って、日本の最新のAI技術トレンドについて教えてください。",
  user_id: "user-12345",
  context: "AIに関するニュース記事やブログから情報を取得してください。",
}, null, 2);

// MUI Theme setup for consistent design
const theme = createTheme({
  palette: {
    primary: {
      main: '#4F46E5', // Indigo-600
    },
    secondary: {
      main: '#10B981', // Emerald-500
    },
  },
  typography: {
    fontFamily: ['Inter', 'sans-serif'].join(','),
  },
});

// Helper function to handle exponential backoff for API calls
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
/**
 * APIレスポンスの構造を定義するインターフェースです。
 * * - text: APIから返される主要なテキストデータ（例: 応答メッセージ）。
 * - response: 追加の構造化データや、APIレスポンスの本体を含むオブジェクト。
 */
export interface ApiResponse {
  // APIからの主要なテキスト応答やステータスサマリー
  text: string;
  
  // RAGシステムからの詳細な回答や、構造化されたデータ（JSON形式）
  response: object;
}
// --- Main Application Component ---
const RAGSender = () => {
  const [queryInput, setQueryInput] = useState<string>(initialQuery);
  const [apiResponse, setApiResponse] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Parse the response for display, prioritizing 'text' or 'response' fields
  const formattedResponse = useMemo(() => {
    if (!apiResponse) return '';
    
    // Attempt to extract the primary text response for cleaner display
    if (typeof apiResponse === 'object' && apiResponse !== null) {
      if (apiResponse.text) return JSON.stringify(apiResponse.text, null, 2);
      if (apiResponse.response) return JSON.stringify(apiResponse.response, null, 2);
    }
    
    // Fallback to pretty-printing the entire JSON object
    return JSON.stringify(apiResponse, null, 2);
  }, [apiResponse]);

  const handleSubmit = useCallback(async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setApiResponse(null);

    let parsedQuery;
    try {
      // 1. Validate and Parse Input JSON
      parsedQuery = JSON.parse(queryInput);
    } catch {
      setError('入力されたクエリが有効なJSON形式ではありません。構文を確認してください。');
      setIsLoading(false);
      return;
    }

    // The API expects the user's JSON object to be nested under the "query" key.
    const requestBody = {
      query: parsedQuery,
    };

    try {
      // 2. Perform the POST Request with backoff
      const response = await exponentialBackoffFetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      // 3. Handle HTTP Errors
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`APIリクエストが失敗しました。ステータスコード: ${response.status}. レスポンス: ${errorText.substring(0, 100)}...`);
      }

      // 4. Handle Successful Response
      const data = await response.json();
      setApiResponse(data);
    } catch (e) {
      console.error('Fetch error:', e);
      // Display user-friendly error message
      setError(`リクエスト処理中にエラーが発生しました: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsLoading(false);
    }
  }, [queryInput]);

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="md" className="py-8 min-h-screen bg-gray-50">
        <Paper elevation={4} className="p-6 md:p-10 rounded-xl">
          <Typography 
            variant="h4" 
            component="h1" 
            gutterBottom 
            className="text-indigo-700 font-extrabold text-center mb-4"
          >
            RAG APIクエリ送信ツール
          </Typography>
          <Typography variant="subtitle1" className="text-gray-600 text-center mb-6">
            指定されたエンドポイントへJSON形式のオブジェクトをPOST送信します。
          </Typography>

          <Box component="form" onSubmit={handleSubmit} className="space-y-6">
            <Typography variant="h6" className="text-gray-700">
              APIエンドポイント: 
              <code className="text-sm text-pink-600 ml-2 p-1 bg-pink-50 rounded">
                {API_ENDPOINT}
              </code>
            </Typography>

            <TextField
              label="JSONクエリ入力 (queryフィールドの内容)"
              multiline
              rows={10}
              fullWidth
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              variant="outlined"
              placeholder='例: {"prompt": "...", "user_id": "..."}'
              helperText="このテキストボックスの内容が、APIリクエストのJSONボディの 'query' キーの値として送信されます。"
              className="bg-white"
              InputProps={{
                style: { fontFamily: 'monospace' },
              }}
            />

            {error && (
              <Alert severity="error" className="shadow-md rounded-lg">
                <Typography component="pre" className="whitespace-pre-wrap">
                  {error}
                </Typography>
              </Alert>
            )}

            <Button
              type="submit"
              variant="contained"
              color="primary"
              size="large"
              fullWidth
              disabled={isLoading || !queryInput.trim()}
              startIcon={isLoading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
              className="py-3 font-bold rounded-lg shadow-lg hover:shadow-xl transition duration-200"
            >
              {isLoading ? '送信中...' : 'APIに送信 (POST)'}
            </Button>
          </Box>

          {(apiResponse || isLoading) && (
            <Box className="mt-8">
              <Divider className="my-6" />
              <Typography variant="h5" gutterBottom className="text-indigo-700 font-bold mb-4">
                APIレスポンス
              </Typography>
              {isLoading ? (
                <Box className="flex justify-center items-center py-8">
                  <CircularProgress />
                </Box>
              ) : (
                <Paper 
                  variant="outlined" 
                  className="p-4 bg-gray-100 rounded-lg overflow-x-auto"
                >
                  <Typography component="pre" className="text-xs sm:text-sm font-mono text-gray-800 whitespace-pre-wrap">
                    {formattedResponse}
                  </Typography>
                </Paper>
              )}
            </Box>
          )}
        </Paper>
      </Container>
    </ThemeProvider>
  );
};

export default RAGSender;
