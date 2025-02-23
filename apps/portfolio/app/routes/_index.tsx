import { Link } from "@remix-run/react";
import { Button } from "~/components/ui/button";
import {
	Card,
	CardHeader,
	CardTitle,
	CardDescription,
} from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { RocketIcon, CodeIcon, PaletteIcon, NotebookIcon } from "lucide-react";

export default function Index() {
	return (
		<div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
			{/* Navigation */}
			<nav className="bg-white/80 backdrop-blur-sm border-b sticky top-0">
				<div className="container mx-auto px-4 py-4 flex items-center justify-between">
					<Link to="/" className="flex items-center gap-2">
						<RocketIcon className="h-6 w-6 text-primary" />
						<span className="text-xl font-bold">Remix Labs</span>
					</Link>

					<div className="hidden md:flex gap-6">
						<Button variant="ghost" asChild>
							<Link to="/blog">Blog</Link>
						</Button>
						<Button variant="ghost" asChild>
							<Link to="/blog">Documentation</Link>
						</Button>
						<Button variant="ghost" asChild>
							<Link to="/blog">Showcase</Link>
						</Button>
					</div>

					<Button className="md:hidden" size="sm" variant="outline">
						Menu
					</Button>
				</div>
			</nav>

			{/* Hero Section */}
			<section className="container mx-auto px-4 py-20 text-center">
				<div className="max-w-3xl mx-auto">
					<h1 className="text-5xl font-bold tracking-tight mb-6">
						Build Amazing Web Experiences
						<span className="text-primary"> with Remix</span>
					</h1>
					<p className="text-xl text-muted-foreground mb-8">
						Accelerate your web development with our curated collection of
						tools, components, and best practices.
					</p>
					<div className="flex gap-4 justify-center">
						<Button size="lg" asChild>
							<Link to="/blog">Get Started</Link>
						</Button>
						<Button size="lg" variant="outline" asChild>
							<Link to="/blog">View Examples</Link>
						</Button>
					</div>
				</div>
			</section>

			{/* Features Grid */}
			<section className="container mx-auto px-4 py-16">
				<div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
					<Card className="hover:shadow-lg transition-shadow">
						<CardHeader>
							<CodeIcon className="h-8 w-8 text-primary mb-4" />
							<CardTitle>Modern Stack</CardTitle>
							<CardDescription>
								TypeScript, Tailwind, Prisma and more
							</CardDescription>
						</CardHeader>
					</Card>

					<Card className="hover:shadow-lg transition-shadow">
						<CardHeader>
							<PaletteIcon className="h-8 w-8 text-primary mb-4" />
							<CardTitle>Beautiful UI</CardTitle>
							<CardDescription>Pre-built accessible components</CardDescription>
						</CardHeader>
					</Card>

					<Card className="hover:shadow-lg transition-shadow">
						<CardHeader>
							<RocketIcon className="h-8 w-8 text-primary mb-4" />
							<CardTitle>Fast & Optimized</CardTitle>
							<CardDescription>
								Built for performance from the start
							</CardDescription>
						</CardHeader>
					</Card>

					<Card className="hover:shadow-lg transition-shadow">
						<CardHeader>
							<NotebookIcon className="h-8 w-8 text-primary mb-4" />
							<CardTitle>Learning Resources</CardTitle>
							<CardDescription>Guides, tutorials and examples</CardDescription>
						</CardHeader>
					</Card>
				</div>
			</section>

			{/* CTA Section */}
			<section className="bg-primary/10 py-20">
				<div className="container mx-auto px-4 text-center">
					<h2 className="text-3xl font-bold mb-6">Start Building Today</h2>
					<p className="text-muted-foreground mb-8 max-w-xl mx-auto">
						Join our community of developers creating amazing web applications
						with modern tools and best practices.
					</p>
					<div className="max-w-md mx-auto flex gap-4">
						<Input placeholder="Enter your email" />
						<Button size="lg">Subscribe</Button>
					</div>
				</div>
			</section>

			{/* Footer */}
			<footer className="border-t bg-slate-50">
				<div className="container mx-auto px-4 py-12">
					<div className="grid md:grid-cols-3 gap-8">
						<div>
							<div className="flex items-center gap-2 mb-4">
								<RocketIcon className="h-6 w-6 text-primary" />
								<span className="font-bold">Remix Labs</span>
							</div>
							<p className="text-sm text-muted-foreground">
								Empowering developers with modern web tools.
							</p>
						</div>

						<div className="grid grid-cols-2 gap-4">
							<div>
								<h3 className="font-semibold mb-3">Resources</h3>
								<ul className="space-y-2 text-sm">
									<li>
										<Link
											to="/blog"
											className="text-muted-foreground hover:text-primary"
										>
											Documentation
										</Link>
									</li>
									<li>
										<Link
											to="/blog"
											className="text-muted-foreground hover:text-primary"
										>
											Blog
										</Link>
									</li>
									<li>
										<Link
											to="/blog"
											className="text-muted-foreground hover:text-primary"
										>
											Showcase
										</Link>
									</li>
								</ul>
							</div>
							<div>
								<h3 className="font-semibold mb-3">Community</h3>
								<ul className="space-y-2 text-sm">
									<li>
										<Link
											to="/blog"
											className="text-muted-foreground hover:text-primary"
										>
											Discord
										</Link>
									</li>
									<li>
										<Link
											to="/blog"
											className="text-muted-foreground hover:text-primary"
										>
											Twitter
										</Link>
									</li>
									<li>
										<Link
											to="/blog"
											className="text-muted-foreground hover:text-primary"
										>
											GitHub
										</Link>
									</li>
								</ul>
							</div>
						</div>

						<div>
							<h3 className="font-semibold mb-3">Newsletter</h3>
							<div className="space-y-4">
								<Label htmlFor="email" className="text-sm">
									Stay updated
								</Label>
								<div className="flex gap-2">
									<Input id="email" placeholder="Email" />
									<Button size="sm">Subscribe</Button>
								</div>
							</div>
						</div>
					</div>

					<div className="mt-8 pt-8 border-t text-center text-sm text-muted-foreground">
						© 2024 Remix Labs. All rights reserved.
					</div>
				</div>
			</footer>
		</div>
	);
}
