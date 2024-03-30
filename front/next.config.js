/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  env: {
    Localhost: "http://175.194.198.155:8080",
  },
  images: {
    domains: ['t1.kakaocdn.net', '175.194.198.155', 'k.kakaocdn.net'], 
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
};

module.exports = nextConfig;
