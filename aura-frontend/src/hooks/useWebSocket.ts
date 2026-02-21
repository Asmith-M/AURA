import { useCallback, useEffect, useRef, useState } from 'react';

export function useWebSocket(url: string, enabled: boolean = true) {
  const [connected, setConnected] = useState<boolean>(false);
  const [messages, setMessages] = useState<unknown[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const sendMessage = useCallback((message: unknown) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    wsRef.current.send(JSON.stringify(message));
  }, []);

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        setMessages((prev) => [...prev, parsed]);
      } catch {
        setMessages((prev) => [...prev, event.data]);
      }
    };

    ws.onerror = () => {
      setConnected(false);
    };

    ws.onclose = () => {
      setConnected(false);
    };

    return () => {
      ws.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [url, enabled]);

  return {
    connected,
    messages,
    sendMessage,
  };
}
