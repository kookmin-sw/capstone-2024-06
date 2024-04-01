"use client";
import React, { useState, useEffect } from "react";
import { useSession } from "next-auth/react";

const SingleClient = ({ userId }: { userId: number }) => {

  const { data: session } = useSession();

  const [message, setMessage] = useState("");

  const [client, setClient] = useState<WebSocket | null>(null);

  useEffect(() => {
    if (!session) return;
    const newClient = new WebSocket(`ws://175.194.198.155:8080/chat/`);
    newClient.onopen = () => {
      newClient.send((session as any)?.access_token)
      console.log(`WebSocket Client Connected for user ${userId}`);
    };
    newClient.onmessage = (message) => {
      const parsedMessage = JSON.parse(message.data);
      
    };
    setClient(newClient);
    
  }, [userId, session]);

  const sendMessage = () => {
    if (message.trim() !== "" && client) {
      client.send(message);
      setMessage("");
    }
  };

  return (
    <div>
      <button onClick={sendMessage}>Send</button>
    </div>
  );
};

const Message = (another:string) => {
  return (
    <div>
      <SingleClient userId={1} />
    </div>
  );
};

export default Message;
