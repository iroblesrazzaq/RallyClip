import { NextResponse, type NextRequest } from "next/server";
import { createClient } from "@/lib/supabase/proxy";

const publicPaths = ["/", "/login", "/auth/callback", "/forgot-password", "/reset-password"];

export async function proxy(request: NextRequest) {
  const { supabase, response } = createClient(request);
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const { pathname } = request.nextUrl;
  const isPublicPath = publicPaths.some((p) =>
    p === "/" ? pathname === "/" : pathname.startsWith(p),
  );

  if (!user && !isPublicPath) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  if (user && pathname === "/login") {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return response;
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
