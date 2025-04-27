import Link from "next/link";
import { Home, ImageIcon, Video, AudioLines } from "lucide-react";
import { ModeToggle } from "./mode.toggle";
import Image from "next/image";
import DarkLogo from "@/public/dark_logo.png"

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 px-5 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className=" flex h-14 items-center">
        <div className="mr-4 flex">
          <Image
            src={DarkLogo}
            width={120}
            height={120}
            alt="Logo"
            className="mr-4"
          />
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link
              href="/"
              className="flex items-center gap-1 transition-colors hover:text-foreground/80"
            >
              <Home className="h-4 w-4" />
              <span>Home</span>
            </Link>
            <Link
              href="/deepfake"
              className="transition-colors hover:text-foreground/80"
            >
              Tools
            </Link>
            <Link
              href="/deepfake/image"
              className="hidden md:flex items-center gap-1 transition-colors hover:text-foreground/80"
            >
              <ImageIcon className="h-4 w-4" />
              <span>Image</span>
            </Link>
            <Link
              href="/deepfake/video"
              className="hidden md:flex items-center gap-1 transition-colors hover:text-foreground/80"
            >
              <Video className="h-4 w-4" />
              <span>Video</span>
            </Link>
            <Link
              href="/deepfake/audio"
              className="hidden md:flex items-center gap-1 transition-colors hover:text-foreground/80"
            >
              <AudioLines className="h-4 w-4" />
              <span>Audio</span>
            </Link>
          </nav>
        </div>
        <div className="flex flex-1 items-center justify-end">
          <nav className="flex items-end">
            <ModeToggle />
          </nav>
        </div>
      </div>
    </header>
  );
}
