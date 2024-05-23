/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["three"],
  reactStrictMode: false,
  env: {

    Localhost: `${process.env.Localhost}`,
    OnlyiP : `${process.env.Localhost}`,
  },
  images: {
    domains: ['t1.kakaocdn.net', `${process.env.Localhosts}`, 'k.kakaocdn.net' , '10.223.115.184'], 

    remotePatterns: [
      {
        protocol: "https",
        hostname: "**", 
      },
    ],
  },
};

module.exports = nextConfig;
