"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, Github } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { LineChart, PieChart } from "@/components/charts";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="mr-4 hidden md:flex">
            <Link href="/" className="mr-6 flex items-center space-x-2">
              <span className="hidden font-bold sm:inline-block">
                DeepfakeDetect
              </span>
            </Link>
            <nav className="flex items-center space-x-6 text-sm font-medium">
              <Link
                href="/"
                className="transition-colors hover:text-foreground/80"
              >
                Home
              </Link>
              <Link
                href="/deepfake"
                className="transition-colors hover:text-foreground/80"
              >
                Tools
              </Link>
              <Link
                href="#about"
                className="transition-colors hover:text-foreground/80"
              >
                About
              </Link>
              <Link
                href="#faq"
                className="transition-colors hover:text-foreground/80"
              >
                FAQ
              </Link>
            </nav>
          </div>
          <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
            <div className="w-full flex-1 md:w-auto md:flex-none">
              <Button asChild className="ml-auto hidden md:flex">
                <Link href="/deepfake">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
          <div className="container px-4 md:px-6">
            <motion.div
              className="flex flex-col items-center space-y-4 text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
                Detect Deepfakes with Confidence
              </h1>
              <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
                Our AI-powered tools help you identify manipulated images,
                videos, and audio with industry-leading accuracy.
              </p>
              <div className="space-x-4">
                <Button asChild size="lg">
                  <Link href="/deepfake">
                    Get Started
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link href="#about">Learn More</Link>
                </Button>
              </div>
            </motion.div>
          </div>
        </section>

        {/* What Are Deepfakes Section */}
        <section
          id="about"
          className="w-full py-12 md:py-24 lg:py-32 bg-muted/50"
        >
          <div className="container px-4 md:px-6">
            <motion.div
              className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                What Are Deepfakes?
              </h2>
              <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                Deepfakes are synthetic media where a person's likeness is
                replaced with someone else's using artificial intelligence.
                These sophisticated manipulations can create convincing but
                fabricated content that poses significant risks to privacy,
                security, and information integrity.
              </p>
              <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                As this technology becomes more accessible, the ability to
                detect deepfakes is increasingly crucial for maintaining trust
                in digital media and protecting individuals and organizations
                from fraud and misinformation.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Tabs Section */}
        <section className="w-full py-12 md:py-24 lg:py-32">
          <div className="container px-4 md:px-6">
            <h2 className="mb-8 text-center text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
              Deepfake Impact Analysis
            </h2>
            <Tabs defaultValue="timeline" className="mx-auto max-w-4xl">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="timeline">Impact Over Time</TabsTrigger>
                <TabsTrigger value="sectors">Affected Sectors</TabsTrigger>
                <TabsTrigger value="incidents">Notable Incidents</TabsTrigger>
              </TabsList>
              <TabsContent value="timeline" className="p-4">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h3 className="mb-4 text-xl font-bold">
                    Financial Losses by Year
                  </h3>
                  <div className="h-[350px] w-full">
                    <LineChart />
                  </div>
                </motion.div>
              </TabsContent>
              <TabsContent value="sectors" className="p-4">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h3 className="mb-4 text-xl font-bold">Affected Sectors</h3>
                  <div className="h-[350px] w-full">
                    <PieChart />
                  </div>
                </motion.div>
              </TabsContent>
              <TabsContent value="incidents" className="p-4">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h3 className="mb-4 text-xl font-bold">Notable Incidents</h3>
                  <Table>
                    <TableCaption>
                      Recent significant deepfake incidents
                    </TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Date</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Impact</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell>Jan 2023</TableCell>
                        <TableCell>Political</TableCell>
                        <TableCell>Election misinformation campaign</TableCell>
                        <TableCell>
                          <Badge variant="destructive">Severe</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Mar 2023</TableCell>
                        <TableCell>Financial</TableCell>
                        <TableCell>CEO voice fraud ($3.2M loss)</TableCell>
                        <TableCell>
                          <Badge variant="destructive">Severe</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Jun 2023</TableCell>
                        <TableCell>Entertainment</TableCell>
                        <TableCell>Celebrity deepfake controversy</TableCell>
                        <TableCell>
                          <Badge variant="warning">Moderate</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Sep 2023</TableCell>
                        <TableCell>Educational</TableCell>
                        <TableCell>Academic credential fraud</TableCell>
                        <TableCell>
                          <Badge variant="warning">Moderate</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Dec 2023</TableCell>
                        <TableCell>Social Media</TableCell>
                        <TableCell>Viral deepfake news story</TableCell>
                        <TableCell>
                          <Badge variant="outline">Resolved</Badge>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </motion.div>
              </TabsContent>
            </Tabs>
          </div>
        </section>

        {/* FAQ Section */}
        <section id="faq" className="w-full py-12 md:py-24 lg:py-32">
          <div className="container px-4 md:px-6">
            <h2 className="mb-8 text-center text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
              Frequently Asked Questions
            </h2>
            <div className="mx-auto max-w-3xl">
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="item-1">
                  <AccordionTrigger>
                    How accurate is deepfake detection?
                  </AccordionTrigger>
                  <AccordionContent>
                    Our detection tools achieve 94-97% accuracy on benchmark
                    datasets. However, as deepfake technology evolves, we
                    continuously update our models to maintain high detection
                    rates.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-2">
                  <AccordionTrigger>
                    Can I detect deepfakes on my smartphone?
                  </AccordionTrigger>
                  <AccordionContent>
                    Yes, our web application is fully responsive and works on
                    smartphones. For optimal performance with large video files,
                    we recommend using a desktop computer.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-3">
                  <AccordionTrigger>
                    How do I report a deepfake?
                  </AccordionTrigger>
                  <AccordionContent>
                    If you've identified a deepfake using our tools, you can
                    report it to the platform where it was shared. Most social
                    media platforms have policies against manipulated media and
                    provide reporting mechanisms.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-4">
                  <AccordionTrigger>
                    Is my uploaded content secure?
                  </AccordionTrigger>
                  <AccordionContent>
                    We prioritize your privacy. All uploaded content is
                    processed securely, not stored permanently, and never shared
                    with third parties. Our analysis happens on encrypted
                    connections and files are deleted after processing.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-5">
                  <AccordionTrigger>
                    What types of deepfakes can be detected?
                  </AccordionTrigger>
                  <AccordionContent>
                    Our tools can detect face-swapped images and videos,
                    synthetic voices, and AI-generated content. We continuously
                    expand our capabilities to address new types of manipulated
                    media as they emerge.
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </div>
        </section>
      </main>

      <footer className="w-full border-t bg-background py-6">
        <div className="container px-4 md:px-6">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            <div className="flex flex-col gap-2">
              <h3 className="text-lg font-semibold">Detection Tools</h3>
              <Link
                href="/deepfake/image"
                className="text-muted-foreground hover:text-foreground"
              >
                Image Detection
              </Link>
              <Link
                href="/deepfake/video"
                className="text-muted-foreground hover:text-foreground"
              >
                Video Detection
              </Link>
              <Link
                href="/deepfake/audio"
                className="text-muted-foreground hover:text-foreground"
              >
                Audio Detection
              </Link>
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="text-lg font-semibold">Resources</h3>
              <Link
                href="#"
                className="text-muted-foreground hover:text-foreground"
              >
                Blog
              </Link>
              <Link
                href="#"
                className="text-muted-foreground hover:text-foreground"
              >
                Research
              </Link>
              <Link
                href="#"
                className="text-muted-foreground hover:text-foreground"
              >
                Documentation
              </Link>
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="text-lg font-semibold">Company</h3>
              <Link
                href="#"
                className="text-muted-foreground hover:text-foreground"
              >
                About Us
              </Link>
              <Link
                href="#"
                className="text-muted-foreground hover:text-foreground"
              >
                Contact
              </Link>
              <Link
                href="#"
                className="text-muted-foreground hover:text-foreground"
              >
                Privacy Policy
              </Link>
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="text-lg font-semibold">Connect</h3>
              <div className="flex space-x-4">
                <Link
                  href="#"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <svg
                    className="h-5 w-5"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      fillRule="evenodd"
                      d="M22 12c0-5.523-4.477-10-10-10S2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.878v-6.987h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.988C18.343 21.128 22 16.991 22 12z"
                      clipRule="evenodd"
                    />
                  </svg>
                </Link>
                <Link
                  href="#"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <svg
                    className="h-5 w-5"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                  </svg>
                </Link>
                <Link
                  href="#"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <svg
                    className="h-5 w-5"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      fillRule="evenodd"
                      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                      clipRule="evenodd"
                    />
                  </svg>
                </Link>
              </div>
            </div>
          </div>
          <div className="mt-8 flex items-center justify-center border-t pt-8">
            <Link
              href="https://github.com"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
            >
              <Github className="h-4 w-4" />
              <span>View on GitHub</span>
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
