"use client";
import React, { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useSearchParams } from "next/navigation";

const SingleClient = ({ userId }: { userId: number }) => {

  const { data: session } = useSession();
  const params = useSearchParams();
  const another = params.get('another');

  const [message, setMessage] = useState("");
  const [receivedMessage, setReceivedMessage] = useState("");
  const [client, setClient] = useState<WebSocket | null>(null);

  useEffect(() => {
    if (!session) return;
    const newClient = new WebSocket(`ws://192.168.127.253:8080/chat/${another}`);
    newClient.onopen = () => {
      newClient.send((session as any)?.access_token)
      console.log(`WebSocket Client Connected for user ${userId}`);
    };
    newClient.onmessage = (message) => {
      const parsemessage = message.data;
      setReceivedMessage(parsemessage);
    };
    setClient(newClient);
    
  }, [userId, session, another]);

  const sendMessage = () => {
    if (message.trim() !== "" && client) {
      client.send(message);
      setMessage("");
    }
  };

  return (
    <div>
      <div>{receivedMessage && <p>{receivedMessage}</p>}</div>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
};

const Message = () => {
  return (
    <div>
      <SingleClient userId={1} />
    </div>
  );
};

export default Message;
