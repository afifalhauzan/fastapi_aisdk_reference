"use client";

import { Button } from "./ui/button";
import { Github } from "lucide-react";
import Link from "next/link";

export const Navbar = () => {
  return (
    <div className="p-2 flex flex-row gap-2 justify-between">
      <Link href="https://github.com/vercel-labs/ai-sdk-preview-python-streaming">
        <Button variant="outline">
          <Github /> View Source Code
        </Button>
      </Link>
    </div>
  );
};
